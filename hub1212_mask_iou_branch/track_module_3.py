import math
import random

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from detectron2.config import configurable
from scipy.optimize import linear_sum_assignment
import fvcore.nn.weight_init as weight_init

from mask2former_video.modeling.transformer_decoder.video_mask2former_transformer_decoder import SelfAttentionLayer,\
    CrossAttentionLayer, FFNLayer, MLP
from .utils.mask_nms import mask_iou, mask_nms


class VideoInstanceSequence(object):
    def __init__(self, start_time: int, matched_gt_id: int = -1, maximum_chache=10):
        self.sT = start_time
        self.eT = -1
        self.maximum_chache = maximum_chache
        self.dead = False
        self.gt_id = matched_gt_id
        self.invalid_frames = 0
        self.embeds = []
        self.pos_embeds = []
        self.pred_logits = []
        self.pred_masks = []
        self.pred_valid = []
        self.appearance = []

        # CTVIS
        self.pos_embeds = []
        self.long_scores = []
        self.similarity_guided_pos_embed = None
        self.similarity_guided_pos_embed_list = []
        self.momentum = 0.75

        self.reid_embeds = []
        self.similarity_guided_reid_embed = None
        self.similarity_guided_reid_embed_list = []

    def update(self, reid_embed):
        self.reid_embeds.append(reid_embed)

        if len(self.similarity_guided_reid_embed_list) == 0:
            self.similarity_guided_reid_embed = reid_embed
            self.similarity_guided_reid_embed_list.append(reid_embed)
        else:
            assert len(self.reid_embeds) > 1
            # Similarity-Guided Feature Fusion
            # https://arxiv.org/abs/2203.14208v1
            all_reid_embed = []
            for embedding in self.reid_embeds[:-1]:
                all_reid_embed.append(embedding)
            all_reid_embed = torch.stack(all_reid_embed, dim=0)

            similarity = torch.sum(torch.einsum("bc,c->b",
                                                F.normalize(all_reid_embed, dim=-1),
                                                F.normalize(reid_embed.squeeze(), dim=-1)
                                                )) / all_reid_embed.shape[0]
            beta = max(0, similarity)
            self.similarity_guided_reid_embed = (1 - beta) * self.similarity_guided_reid_embed + beta * reid_embed
            self.similarity_guided_reid_embed_list.append(self.similarity_guided_reid_embed)

        if len(self.reid_embeds) > self.maximum_chache:
            self.reid_embeds.pop(0)

    def update_pos(self, pos_embed):
        self.pos_embeds.append(pos_embed)

        if len(self.similarity_guided_pos_embed_list) == 0:
            self.similarity_guided_pos_embed = pos_embed
            self.similarity_guided_pos_embed_list.append(pos_embed)
        else:
            assert len(self.pos_embeds) > 1
            # Similarity-Guided Feature Fusion
            # https://arxiv.org/abs/2203.14208v1
            all_pos_embed = []
            for embedding in self.pos_embeds[:-1]:
                all_pos_embed.append(embedding)
            all_pos_embed = torch.stack(all_pos_embed, dim=0)

            similarity = torch.sum(torch.einsum("bc,c->b",
                                                F.normalize(all_pos_embed, dim=-1),
                                                F.normalize(pos_embed.squeeze(), dim=-1)
                                                )) / all_pos_embed.shape[0]

            # TODO, using different similarity function
            beta = max(0, similarity)
            self.similarity_guided_pos_embed = (1 - beta) * self.similarity_guided_pos_embed + beta * pos_embed
            self.similarity_guided_pos_embed_list.append(self.similarity_guided_pos_embed)

        if len(self.pos_embeds) > self.maximum_chache:
            self.pos_embeds.pop(0)


class VideoInstanceCutter(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 256,
        feedforward_dim: int = 2048,
        num_head: int = 8,
        decoder_layer_num: int = 6,
        mask_dim: int = 256,
        num_classes: int = 25,
        num_new_ins: int = 100,

        training_select_threshold: float = 0.1,
        inference_select_threshold: float = 0.1,
        kick_out_frame_num: int = 8,
        mask_nms_thr: float = 0.5,
        # reid
        reid_hidden_dim: int = 256,
        num_reid_head_layers: int = 3,
        match_type: str = 'greedy',
        match_score_thr: float = 0.3,
    ):
        super().__init__()

        self.num_heads = num_head
        self.hidden_dim = hidden_dim
        self.num_layers = decoder_layer_num
        self.num_classes = num_classes
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=feedforward_dim,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.pos_embed = MLP(mask_dim, hidden_dim, hidden_dim, 3)
        if num_reid_head_layers > 0:
            self.reid_embed = MLP(
                hidden_dim, reid_hidden_dim, hidden_dim, num_reid_head_layers)
            for layer in self.reid_embed.layers:
                weight_init.c2_xavier_fill(layer)
        else:
            self.reid_embed = torch.nn.Identity()  # do nothing
        self.num_reid_head_layers = num_reid_head_layers

        # mask features projection
        self.mask_feature_proj = nn.Conv2d(
            mask_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.new_ins_embeds = nn.Embedding(1, hidden_dim)
        self.disappear_embed = nn.Embedding(1, hidden_dim)
        # self.disappear_norm = nn.LayerNorm(hidden_dim)
        # self.disappear_ffn = FFNLayer(hidden_dim, feedforward_dim)

        # record previous frame information
        self.last_seq_ids = None
        self.track_queries = None
        self.track_embeds = None
        self.cur_disappear_embeds = None
        self.prev_frame_indices = None
        self.tgt_ids_for_track_queries = None
        self.disappear_fq_mask = None
        self.disappear_tgt_id = None
        self.disappear_trcQ_id = None
        self.disappeared_tgt_ids = []
        self.new_born_tgt_ids = []
        self.video_ins_hub = dict()
        self.gt_ins_hub = dict()

        self.num_new_ins = num_new_ins
        self.training_select_thr = training_select_threshold
        self.inference_select_thr = inference_select_threshold
        self.mask_nms_thr = mask_nms_thr
        self.kick_out_frame_num = kick_out_frame_num
        self.match_type = match_type
        self.match_score_thr = match_score_thr

    def _clear_memory(self):
        del self.video_ins_hub
        self.video_ins_hub = dict()
        self.gt_ins_hub = dict()
        self.last_seq_ids = None
        self.track_queries = None
        self.track_embeds = None
        self.cur_disappear_embeds = None
        self.prev_frame_indices = None
        self.tgt_ids_for_track_queries = None
        self.disappear_fq_mask = None
        self.disappear_tgt_id = None
        self.disappeared_tgt_ids = []
        self.new_born_tgt_ids = []
        self.disappear_trcQ_id = None
        return

    def readout(self, read_type: str = "last"):
        assert read_type in ["last", "last_valid", "last_pos", "last_valid_pos"]

        if read_type == "last":
            out_embeds = []
            for seq_id in self.last_seq_ids:
                out_embeds.append(self.video_ins_hub[seq_id].embeds[-1])
            if len(out_embeds):
                return torch.stack(out_embeds, dim=0).unsqueeze(1)  # q, 1, c
            else:
                return torch.empty(size=(0, 1, self.hidden_dim), dtype=torch.float32).to("cuda")
        elif read_type == "last_pos":
            out_pos_embeds = []
            for seq_id in self.last_seq_ids:
                out_pos_embeds.append(self.video_ins_hub[seq_id].similarity_guided_pos_embed)
            if len(out_pos_embeds):
                return torch.stack(out_pos_embeds, dim=0).unsqueeze(1)  # q, 1, c
            else:
                return torch.empty(size=(0, 1, self.hidden_dim), dtype=torch.float32).to("cuda")
        else:
            raise NotImplementedError

    def forward(self, frame_embeds_no_norm, mask_features, targets, frames_info, matcher,
                resume=False, using_thr=False):
        ori_mask_features = mask_features
        mask_features_shape = mask_features.shape
        mask_features = self.mask_feature_proj(mask_features.flatten(0, 1)).reshape(*mask_features_shape)  # (b, t, c, h, w)

        frame_embeds_no_norm = frame_embeds_no_norm.permute(2, 3, 0, 1)  # t, q, b, c
        T, fQ, B, _ = frame_embeds_no_norm.shape
        assert B == 1
        all_outputs = []

        new_ins_embeds = self.new_ins_embeds.weight.unsqueeze(1).repeat(self.num_new_ins, B, 1)  # nq, b, c
        disappear_embed = self.disappear_embed.weight.unsqueeze(1).repeat(1, B, 1)  # 1, b, c
        for i in range(T):
            ms_output = []
            single_frame_embeds_no_norm = frame_embeds_no_norm[i]  # q, b, c
            targets_i = targets[i].copy()
            valid_fq_mask = frames_info["valid"][i][0]
            num_valid_fq = valid_fq_mask.sum()
            # the first frame of a video
            if i == 0 and resume is False:
                self._clear_memory()
                ms_output.append(single_frame_embeds_no_norm)
                for j in range(self.num_layers):
                    output = self.transformer_cross_attention_layers[j](
                        ms_output[-1], single_frame_embeds_no_norm
                    )
                    output = self.transformer_self_attention_layers[j](output)
                    output = self.transformer_ffn_layers[j](output)
                    ms_output.append(output)
            else:
                disappear_fq_mask = torch.zeros(size=(fQ,), dtype=torch.bool).to("cuda")
                if self.tgt_ids_for_track_queries is not None and len(self.tgt_ids_for_track_queries) > 2:
                    select_idx = random.randrange(0, self.tgt_ids_for_track_queries.shape[0])
                    select_tgt_id = self.tgt_ids_for_track_queries[select_idx]
                    if (not using_thr) or len(self.disappeared_tgt_ids) > 0:
                        # each sample (5 frames) only model disappearance once
                        self.disappear_tgt_id = None
                        self.disappear_trcQ_id = None
                    elif select_tgt_id != -1 and select_tgt_id in frames_info["indices"][i][0][1]:
                        aux_tgt_i_for_each_fq = frames_info["aux_indices"][i][0][1]
                        disappear_fq_mask[aux_tgt_i_for_each_fq == select_tgt_id] = True
                        self.disappear_tgt_id = select_tgt_id
                        self.disappear_trcQ_id = select_idx
                        self.disappeared_tgt_ids.append(select_tgt_id)
                    else:
                        self.disappear_tgt_id = None
                        self.disappear_trcQ_id = None
                else:
                    self.disappear_tgt_id = None
                    self.disappear_trcQ_id = None
                self.disappear_fq_mask = disappear_fq_mask
                valid_appear_fq_mask = valid_fq_mask & (~disappear_fq_mask)

                detQ_pos = self.get_mask_pos_embed(frames_info["pred_masks"][i][0][None], ori_mask_features[:, i, ...])
                pilot = torch.cat([self.track_queries, new_ins_embeds], dim=0)
                pilot_pos = torch.cat([self.track_embeds, detQ_pos], dim=0)
                disappear_embeds = disappear_embed.repeat(self.track_queries.shape[0], 1, 1)
                attn_mask = torch.zeros(size=(B, self.num_heads, pilot.shape[0], fQ + self.track_queries.shape[0]),
                                        dtype=torch.bool).to("cuda")
                attn_mask[:, :, :self.track_queries.shape[0], :fQ] = \
                    ~valid_appear_fq_mask[None, None, None, :].repeat(B, self.num_heads, self.track_queries.shape[0], 1)
                attn_mask[:, :, self.track_queries.shape[0]:, fQ:] = True
                attn_mask = attn_mask.flatten(0, 1)

                ms_output.append(pilot)
                for j in range(self.num_layers):
                    output = self.transformer_cross_attention_layers[j](
                        ms_output[-1], torch.cat([single_frame_embeds_no_norm, disappear_embeds], dim=0),
                        query_pos=pilot_pos,
                        pos=torch.cat([detQ_pos, self.track_queries], dim=0),
                        memory_mask=attn_mask
                    )
                    output = self.transformer_self_attention_layers[j](output)
                    output = self.transformer_ffn_layers[j](output)
                    ms_output.append(output)

            ms_output = torch.stack(ms_output, dim=0)  # (num_layers+1, q, b, c)
            outputs_class, outputs_mask = self.prediction(ms_output, mask_features[:, i, ...])
            out_dict = {
                "pred_logits": outputs_class[-1],  # b, q, k+1
                "pred_masks": outputs_mask[-1],  # b, q, h, w
                "num_new_ins": num_valid_fq.item()
            }

            if i == 0 and resume is False:
                out_dict.update({
                    "indices": frames_info["indices"][i]  # [(src_idx, tgt_idx)], only valid inst
                })

                if using_thr:
                    # tgt_ids_for_each_query = tgt_ids_for_all_fq[valid_fq_mask]
                    tgt_ids_for_each_query = torch.full(size=(ms_output.shape[1],), dtype=torch.int64,
                                                        fill_value=-1).to("cuda")
                    tgt_ids_for_each_query[frames_info["indices"][i][0][0]] = frames_info["indices"][i][0][1]
                    pred_scores = torch.max(outputs_class[-1, 0].softmax(-1)[:, :-1], dim=-1)[0]
                    valid_track_query = pred_scores > self.training_select_thr
                else:
                    tgt_ids_for_each_query = torch.full(size=(ms_output.shape[1],), dtype=torch.int64,
                                                        fill_value=-1).to("cuda")
                    tgt_ids_for_each_query[frames_info["indices"][i][0][0]] = frames_info["indices"][i][0][1]
                    select_track_queries = torch.rand(size=(len(frames_info["indices"][i][0][0]),),
                                                      dtype=torch.float32).to("cuda") > 0.5
                    kick_out_src_indices = frames_info["indices"][i][0][0][select_track_queries]
                    # tgt_ids_for_all_fq[kick_out_src_indices] = -1
                    tgt_ids_for_each_query[kick_out_src_indices] = -1
                    # tgt_ids_for_each_query = tgt_ids_for_all_fq[valid_fq_mask]
                    disappearance_mask = torch.zeros(size=(ms_output.shape[1],), dtype=torch.bool).to("cuda")
                    # disappearance_mask = torch.zeros(size=(fQ,), dtype=torch.bool).to("cuda")
                    disappearance_mask[kick_out_src_indices] = True
                    # valid_track_query = ~disappearance_mask[valid_fq_mask]
                    valid_track_query = ~disappearance_mask
            else:
                indices = matcher(out_dict, targets_i, self.prev_frame_indices)
                out_dict.update({
                    "indices": indices
                })

                if using_thr:
                    tgt_ids_for_each_query = torch.full(size=(ms_output.shape[1],), dtype=torch.int64,
                                                        fill_value=-1).to("cuda")
                    tgt_ids_for_each_query[indices[0][0]] = indices[0][1]
                    pred_scores = torch.max(outputs_class[-1, 0].softmax(-1)[:, :-1], dim=-1)[0]
                    valid_track_query = pred_scores > self.training_select_thr
                else:
                    tgt_ids_for_each_query = torch.full(size=(ms_output.shape[1],), dtype=torch.int64,
                                                        fill_value=-1).to("cuda")
                    tgt_ids_for_each_query[indices[0][0]] = indices[0][1]
                    valid_track_query = torch.ones(size=(ms_output.shape[1],)).to("cuda") < 0
                    valid_track_query[indices[0][0]] = True

            if not using_thr:
                select_query_tgt_ids = tgt_ids_for_each_query[valid_track_query]  # q',

                self.track_queries = ms_output[-1][valid_track_query]  # q', b, c
                prev_src_indices = torch.nonzero(select_query_tgt_ids + 1).squeeze(-1)
                prev_tgt_indices = torch.index_select(select_query_tgt_ids, dim=0, index=prev_src_indices)
                self.prev_frame_indices = (prev_src_indices, prev_tgt_indices)

                self.track_embeds = self.get_mask_pos_embed(outputs_mask[-1, :, valid_track_query, :, :],
                                                            ori_mask_features[:, i, ...])  # q', b, c
                out_dict.update({
                    "aux_outputs": self._set_aux_loss(outputs_class, outputs_mask),
                    "disappear_tgt_id": -10000 if self.disappear_tgt_id is None else self.disappear_tgt_id,
                })
                all_outputs.append(out_dict)
                continue

            self.track_queries = ms_output[-1][valid_track_query]  # q', b, c
            select_query_tgt_ids = tgt_ids_for_each_query[valid_track_query]  # q',
            prev_src_indices = torch.nonzero(select_query_tgt_ids + 1).squeeze(-1)
            prev_tgt_indices = torch.index_select(select_query_tgt_ids, dim=0, index=prev_src_indices)
            self.prev_frame_indices = (prev_src_indices, prev_tgt_indices)
            self.tgt_ids_for_track_queries = tgt_ids_for_each_query[valid_track_query]

            track_embeds = self.get_mask_pos_embed(outputs_mask[-1, ...],
                                                   ori_mask_features[:, i, ...])  # q', b, c
            cur_seq_ids = []
            for k, valid in enumerate(valid_track_query):
                if self.last_seq_ids is not None and k < len(self.last_seq_ids):
                    seq_id = self.last_seq_ids[k]
                else:
                    seq_id = random.randint(0, 100000)
                    while seq_id in self.video_ins_hub:
                        seq_id = random.randint(0, 100000)
                    assert not seq_id in self.video_ins_hub
                if valid:
                    if not seq_id in self.video_ins_hub:
                        self.video_ins_hub[seq_id] = VideoInstanceSequence(0, tgt_ids_for_each_query[k])
                    self.video_ins_hub[seq_id].update_pos(track_embeds[k, 0, :])
                    cur_seq_ids.append(seq_id)
            self.last_seq_ids = cur_seq_ids
            self.track_embeds = self.readout("last_pos")

            out_dict.update({
                "aux_outputs": self._set_aux_loss(outputs_class, outputs_mask),
                "disappear_tgt_id": -10000 if self.disappear_tgt_id is None else self.disappear_tgt_id,
            })
            all_outputs.append(out_dict)

        return all_outputs

    def inference(self, frame_embeds_no_norm, mask_features, frames_info, start_frame_id, resume=False, to_store="cpu"):
        ori_mask_features = mask_features
        mask_features_shape = mask_features.shape
        mask_features = self.mask_feature_proj(mask_features.flatten(0, 1)).reshape(
            *mask_features_shape)  # (b, t, c, h, w)

        frame_embeds_no_norm = frame_embeds_no_norm.permute(2, 3, 0, 1)  # t, q, b, c
        T, fQ, B, _ = frame_embeds_no_norm.shape
        assert B == 1

        new_ins_embeds = self.new_ins_embeds.weight.unsqueeze(1).repeat(self.num_new_ins, B, 1)  # nq, b, c
        disappear_embed = self.disappear_embed.weight.unsqueeze(1).repeat(1, B, 1)  # 1, b, c
        for i in range(T):
            ms_output = []
            single_frame_embeds_no_norm = frame_embeds_no_norm[i]  # q, b, c
            valid_fq_mask = frames_info["valid"][i][0]
            num_valid_fq = valid_fq_mask.sum()
            # the first frame of a video
            if i == 0 and resume is False:
                self._clear_memory()
                ms_output.append(single_frame_embeds_no_norm)
                for j in range(self.num_layers):
                    output = self.transformer_cross_attention_layers[j](
                        ms_output[-1], single_frame_embeds_no_norm
                    )
                    output = self.transformer_self_attention_layers[j](output)
                    output = self.transformer_ffn_layers[j](output)
                    ms_output.append(output)
            else:
                detQ_pos = self.get_mask_pos_embed(frames_info["pred_masks"][i][0][None], ori_mask_features[:, i, ...])
                pilot = torch.cat([self.track_queries, new_ins_embeds], dim=0)
                pilot_pos = torch.cat([self.track_embeds, detQ_pos], dim=0)
                disappear_embeds = disappear_embed.repeat(self.track_queries.shape[0], 1, 1)
                attn_mask = torch.zeros(size=(B, self.num_heads, pilot.shape[0], fQ + self.track_queries.shape[0]),
                                        dtype=torch.bool).to("cuda")
                attn_mask[:, :, :self.track_queries.shape[0], :fQ] = \
                    ~valid_fq_mask[None, None, None, :].repeat(B, self.num_heads, self.track_queries.shape[0], 1)
                attn_mask[:, :, self.track_queries.shape[0]:, fQ:] = True
                attn_mask = attn_mask.flatten(0, 1)

                ms_output.append(pilot)
                for j in range(self.num_layers):
                    output = self.transformer_cross_attention_layers[j](
                        ms_output[-1], torch.cat([single_frame_embeds_no_norm, disappear_embeds], dim=0),
                        query_pos=pilot_pos,
                        pos=torch.cat([detQ_pos, self.track_queries], dim=0),
                        memory_mask=attn_mask
                    )
                    output = self.transformer_self_attention_layers[j](output)
                    output = self.transformer_ffn_layers[j](output)
                    ms_output.append(output)

            ms_output = torch.stack(ms_output, dim=0)  # (num_layers+1, q, b, c)
            outputs_class, outputs_mask = self.prediction(ms_output[-1:], mask_features[:, i, ...])

            track_embeds = self.get_mask_pos_embed(outputs_mask[-1, ...], ori_mask_features[:, i, ...])  # q', b, c

            cur_seq_ids = []
            pred_scores = torch.max(outputs_class[-1, 0].softmax(-1)[:, :-1], dim=1)[0]
            valid_queries = pred_scores > self.inference_select_thr
            for k, valid in enumerate(valid_queries):
                if self.last_seq_ids is not None and k < len(self.last_seq_ids):
                    seq_id = self.last_seq_ids[k]
                else:
                    seq_id = random.randint(0, 100000)
                    while seq_id in self.video_ins_hub:
                        seq_id = random.randint(0, 100000)
                    assert not seq_id in self.video_ins_hub
                if valid:
                    if not seq_id in self.video_ins_hub:
                        self.video_ins_hub[seq_id] = VideoInstanceSequence(start_frame_id + i, seq_id)
                    self.video_ins_hub[seq_id].embeds.append(ms_output[-1, k, 0, :])
                    self.video_ins_hub[seq_id].pred_logits.append(outputs_class[-1, 0, k, :])
                    self.video_ins_hub[seq_id].pred_masks.append(
                        outputs_mask[-1, 0, k, ...].to(to_store).to(torch.float32))
                    self.video_ins_hub[seq_id].invalid_frames = 0
                    self.video_ins_hub[seq_id].appearance.append(True)

                    self.video_ins_hub[seq_id].update_pos(track_embeds[k, 0, :])

                    cur_seq_ids.append(seq_id)
                elif self.last_seq_ids is not None and seq_id in self.last_seq_ids:
                    self.video_ins_hub[seq_id].invalid_frames += 1
                    if self.video_ins_hub[seq_id].invalid_frames >= self.kick_out_frame_num:
                        self.video_ins_hub[seq_id].dead = True
                        continue
                    self.video_ins_hub[seq_id].embeds.append(ms_output[-1, k, 0, :])
                    self.video_ins_hub[seq_id].pred_logits.append(outputs_class[-1, 0, k, :])
                    self.video_ins_hub[seq_id].pred_masks.append(
                        outputs_mask[-1, 0, k, ...].to(to_store).to(torch.float32))
                    # self.video_ins_hub[seq_id].pred_masks.append(
                    #     torch.zeros_like(outputs_mask[-1, 0, k, ...]).to(to_store).to(torch.float32))
                    self.video_ins_hub[seq_id].appearance.append(False)

                    cur_seq_ids.append(seq_id)
            self.last_seq_ids = cur_seq_ids
            self.track_queries = self.readout("last")
            self.track_embeds = self.readout("last_pos")

    def prediction(self, outputs, mask_features):
        # outputs (l, q, b, c)
        # mask_features (b, c, h, w)
        decoder_output = self.decoder_norm(outputs.transpose(1, 2))
        outputs_class = self.class_embed(decoder_output)  # l, b, q, k+1
        mask_embed = self.mask_embed(decoder_output)      # l, b, q, c
        outputs_mask = torch.einsum("lbqc,bchw->lbqhw", mask_embed, mask_features)

        return outputs_class, outputs_mask

    def get_mask_pos_embed(self, mask, mask_features):
        """
        mask: b, q, h, w
        mask_features: b, c, h, w
        """
        pos_embeds_list = []
        num_chunk = mask.shape[1] // 50 + 1
        for i in range(num_chunk):
            start = i * 50
            end = start + 50 if start + 50 < mask.shape[1] else mask.shape[1]

            seg_mask = (mask[:, start:end, :, :].sigmoid() > 0.5).to("cuda")
            mask_feats = seg_mask[:, :, None, :, :] * mask_features[:, None, ...]  # b, q, c, h, w
            pos_embeds = torch.sum(mask_feats.flatten(3, 4), dim=-1) / (
                    torch.sum(seg_mask.flatten(2, 3), dim=-1, keepdim=True) + 1e-8)
            pos_embeds = self.pos_embed(pos_embeds)
            pos_embeds_list.append(pos_embeds.transpose(0, 1))

        return torch.cat(pos_embeds_list, dim=0)

    def filter_new_ins(self, pred_logits, pred_masks):
        """
        pred_logits: q'+nq, k+1
        pred_masks: q"+nq, h, w
        """
        num_instances = pred_masks.shape[0]
        pred_scores = torch.max(pred_logits.softmax(-1)[:, :-1], dim=1)[0]
        seg_masks = pred_masks[:, None].sigmoid() > 0.5

        trcQ_scores = pred_scores[:-self.num_new_ins]
        trcQ_seg_masks = seg_masks[:-self.num_new_ins]
        _, sorted_indices = torch.sort(trcQ_scores, descending=True)
        trcQ_scores = trcQ_scores[sorted_indices]
        trcQ_seg_masks = trcQ_seg_masks[sorted_indices]
        trcQ_keep = mask_nms(trcQ_seg_masks, trcQ_scores, self.mask_nms_thr)
        keep = [True for _ in range(num_instances)]
        for i, idx in enumerate(sorted_indices):
            keep[idx] = trcQ_keep[i]

        for i in range(0, num_instances - self.num_new_ins):
            if pred_scores[i] < self.inference_select_thr:
                keep[i] = False
                continue
            mask_i = seg_masks[i]
            for j in range(num_instances - self.num_new_ins, num_instances):
                if not keep[j]:
                    continue
                if pred_scores[j] < self.inference_select_thr:
                    keep[j] = False
                    continue
                mask_j = seg_masks[j]
                iou = mask_iou(mask_i, mask_j)[0]
                if iou > self.mask_nms_thr:
                    keep[j] = False
        keep = torch.as_tensor(keep).to(torch.bool).to("cuda")
        return keep

    @torch.jit.unused
    def _set_aux_loss(self, outputs_cls, outputs_mask):
        return [{"pred_logits": a,
                 "pred_masks": b,
                 "disappear_tgt_id": -10000 if self.disappear_tgt_id is None else self.disappear_tgt_id,
                 } for a, b
                in zip(outputs_cls[:-1], outputs_mask[:-1])]