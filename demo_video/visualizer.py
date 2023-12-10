# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# reference: https://github.com/sukjunhwang/IFC/blob/master/projects/IFC/demo/visualizer.py
import torch
import numpy as np
import matplotlib.colors as mplc

from detectron2.utils.visualizer import ColorMode, GenericMask, Visualizer, _create_text_labels


_ID_JITTERS = [[0.9047944201469568, 0.3241718265806123, 0.33443746665210006], [0.4590171386127151, 0.9095038146383864, 0.3143840671974788], [0.4769356899795538, 0.5044406738441948, 0.5354530846360839], [0.00820945625670777, 0.24099210193126785, 0.15471834055332978], [0.6195684374237388, 0.4020380013509799, 0.26100266066404676], [0.08281237756545068, 0.05900744492710419, 0.06106221202154216], [0.2264886829978755, 0.04925271007292076, 0.10214429345996079], [0.1888247470009874, 0.11275000298612425, 0.46112894830685514], [0.37415767691880975, 0.844284596118331, 0.950471611180866], [0.3817344218157631, 0.3483259270707101, 0.6572989333690541], [0.2403115731054466, 0.03078280287279167, 0.5385975692534737], [0.7035076951650824, 0.12352084932325424, 0.12873080308790197], [0.12607434914489934, 0.111244793010015, 0.09333334699716023], [0.6551607300342269, 0.7003064103554443, 0.4131794512286162], [0.13592107365596595, 0.5390702818232149, 0.004540643174930525], [0.38286244894454347, 0.709142545393449, 0.529074791609835], [0.4279376583651734, 0.5634708596431771, 0.8505569717104301], [0.3460488523902999, 0.464769595519293, 0.6676839675477276], [0.8544063246675081, 0.5041190233407755, 0.9081217697141578], [0.9207009090747208, 0.2403865944739051, 0.05375410999863772], [0.6515786136947107, 0.6299918449948327, 0.45292029442034387], [0.986174217295693, 0.2424849846977214, 0.3981993323108266], [0.22101915872994693, 0.3408589198278038, 0.006381420347677524], [0.3159785813515982, 0.1145748921741011, 0.595754317197274], [0.10263421488052715, 0.5864139253490858, 0.23908000741142432], [0.8272999391532938, 0.6123527260897751, 0.3365197327803193], [0.5269583712937912, 0.25668929554516506, 0.7888411215078127], [0.2433880265410031, 0.7240751234287827, 0.8483215810528648], [0.7254601709704898, 0.8316525547295984, 0.9325253855921963], [0.5574483824856672, 0.2935331727879944, 0.6594839453793155], [0.6209642371433579, 0.054030693198821256, 0.5080873988178534], [0.9055507077365624, 0.12865888619203514, 0.9309191861440005], [0.9914469722960537, 0.3074114506206205, 0.8762107657323488], [0.4812682518247371, 0.15055826298548158, 0.9656340505308308], [0.6459219454316445, 0.9144794010251625, 0.751338812155106], [0.860840174209798, 0.8844626353077639, 0.3604624506769899], [0.8194991672032272, 0.926399617787601, 0.8059222327343247], [0.6540413175393658, 0.04579445254618297, 0.26891917826531275], [0.37778835833987046, 0.36247927666109536, 0.7989799305827889], [0.22738304978177726, 0.9038018263773739, 0.6970838854138303], [0.6362015495896184, 0.527680794236961, 0.5570915425178721], [0.6436401915860954, 0.6316925317144524, 0.9137151236993912], [0.04161828388587163, 0.3832413349082706, 0.6880829921949752], [0.7768167825719299, 0.8933821497682587, 0.7221278391266809], [0.8632760876301346, 0.3278628094906323, 0.8421587587114462], [0.8556499133262127, 0.6497385872901932, 0.5436895688477963], [0.9861940318610894, 0.03562313777386272, 0.9183454677106616], [0.8042586091176366, 0.6167222703170994, 0.24181981557207644], [0.9504247117633057, 0.3454233714011461, 0.6883727005547743], [0.9611909135491202, 0.46384154263898114, 0.32700443315058914], [0.523542176970206, 0.446222414615845, 0.9067402987747814], [0.7536954008682911, 0.6675512338797588, 0.22538238957839196], [0.1554052265688285, 0.05746097492966129, 0.8580358872587424], [0.8540838640971405, 0.9165504335482566, 0.6806982829158964], [0.7065090319405029, 0.8683059983962002, 0.05167128320624026], [0.39134812961899124, 0.8910075505622979, 0.7639815712623922], [0.1578117311479783, 0.20047326898284668, 0.9220177338840568], [0.2017488993096358, 0.6949259970936679, 0.8729196864798128], [0.5591089340651949, 0.15576770423813258, 0.1469857469387812], [0.14510398622626974, 0.24451497734532168, 0.46574271993578786], [0.13286397822351492, 0.4178244533944635, 0.03728728952131943], [0.556463206310225, 0.14027595183361663, 0.2731537988657907], [0.4093837966398032, 0.8015225687789814, 0.8033567296903834], [0.527442563956637, 0.902232617214431, 0.7066626674362227], [0.9058355503297827, 0.34983989180213004, 0.8353262183839384], [0.7108382186953104, 0.08591307895133471, 0.21434688012521974], [0.22757345065207668, 0.7943075496583976, 0.2992305547627421], [0.20454109788173636, 0.8251670332103687, 0.012981987094547232], [0.7672562637297392, 0.005429019973062554, 0.022163616037108702], [0.37487345910117564, 0.5086240194440863, 0.9061216063654387], [0.9878004014101087, 0.006345852772772331, 0.17499753379350858], [0.030061528704491303, 0.1409704315546606, 0.3337131835834506], [0.5022506782611504, 0.5448435505388706, 0.40584238936140726], [0.39560774627423445, 0.8905943695833262, 0.5850815030921116], [0.058615671926786406, 0.5365713844300387, 0.1620457551256279], [0.41843842882069693, 0.1536005983609976, 0.3127878501592438], [0.05947621790155899, 0.5412421167331932, 0.2611322146455659], [0.5196159938235607, 0.7066461551682705, 0.970261497412556], [0.30443031606149007, 0.45158581060034975, 0.4331841153149706], [0.8848298403933996, 0.7241791700943656, 0.8917110054596072], [0.5720260591898779, 0.3072801598203052, 0.8891066705989902], [0.13964015336177327, 0.2531778096760302, 0.5703756837403124], [0.2156307542329836, 0.4139947500641685, 0.87051676884144], [0.10800455881891169, 0.05554646035458266, 0.2947027428551443], [0.35198009410633857, 0.365849666213808, 0.06525787683513773], [0.5223264108118847, 0.9032195574351178, 0.28579084943315025], [0.7607724246546966, 0.3087194381828555, 0.6253235528354899], [0.5060485442077824, 0.19173600467625274, 0.9931175692203702], [0.5131805830323746, 0.07719515392040577, 0.923212006754969], [0.3629762141280106, 0.02429179642710888, 0.6963754952399983], [0.7542592485456767, 0.6478893299494212, 0.3424965345400731], [0.49944574453364454, 0.6775665366832825, 0.33758796076989583], [0.010621818120767679, 0.8221571611173205, 0.5186257457566332], [0.5857910304290109, 0.7178133992025467, 0.9729243483606071], [0.16987399482717613, 0.9942570210657463, 0.18120758122552927], [0.016362572521240848, 0.17582788603087263, 0.7255176922640298], [0.10981764283706419, 0.9078582203470377, 0.7638063718334003], [0.9252097840441119, 0.3330197086990039, 0.27888705301420136], [0.12769972651171546, 0.11121470804891687, 0.12710743734391716], [0.5753520518360334, 0.2763862879599456, 0.6115636613363361]]


class TrackVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE, id_memories=None):
        super().__init__(
            img_rgb, metadata=metadata, scale=scale, instance_mode=instance_mode
        )
        self.cpu_device = torch.device("cpu")

        if id_memories is None:
            self.id_memories = {}
        else:
            self.id_memories = id_memories
    
    def _jitter(self, color, id):
        """
        Randomly modifies given color to produce a slightly different color than the color given.
        Args:
            color (tuple[double]): a tuple of 3 elements, containing the RGB values of the color
                picked. The values in the list are in the [0.0, 1.0] range.
        Returns:
            jittered_color (tuple[double]): a tuple of 3 elements, containing the RGB values of the
                color after being jittered. The values in the list are in the [0.0, 1.0] range.
        """
        # id = id // len(_ID_JITTERS)
        id = id % len(_ID_JITTERS)
        color = mplc.to_rgb(color)
        vec = _ID_JITTERS[id]
        # better to do it in another color space
        vec = vec / np.linalg.norm(vec) * 0.5
        res = np.clip(vec + color, 0, 1)
        return tuple(res)

    def _get_continuous_id(self, id):
        if id in self.id_memories.keys():
            return self.id_memories[id]
        else:
            self.id_memories[id] = len(self.id_memories)
            return self.id_memories[id]

    def draw_instance_predictions(self, predictions, ids=None):
        """
        Draw instance-level prediction results on an image.
        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
        Returns:
            output (VisImage): image object with visualizations.
        """
        preds = predictions.to(self.cpu_device)

        boxes = preds.pred_boxes if preds.has("pred_boxes") else None
        scores = preds.scores if preds.has("scores") else None
        classes = preds.pred_classes if preds.has("pred_classes") else None

        thing_classes = self.metadata.get("thing_classes", None)
        stuff_classes = self.metadata.get("stuff_classes", None)
        if stuff_classes is None:
            dataset_classes = thing_classes
        else:
            if thing_classes is None:
                dataset_classes = stuff_classes
            else:
                dataset_classes = thing_classes + stuff_classes
        labels = _create_text_labels(classes, scores, dataset_classes)
        if labels is not None:
            if ids is None:
                labels = ["[{}] ".format(_id) + l for _id, l in enumerate(labels)]
            else:
                labels = ["[{}] ".format(self._get_continuous_id(_id)) + l for _id, l in zip(ids, labels)]

        if preds.has("pred_masks"):
            masks = np.asarray(preds.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if classes is None:
            return self.output

        thing_colors = self.metadata.get("thing_colors", None)
        stuff_colors = self.metadata.get("stuff_colors", None)
        if stuff_classes is None:
            dataset_colors = thing_colors
        else:
            if thing_colors is None:
                dataset_colors = stuff_colors
            else:
                dataset_colors = thing_colors + stuff_colors

        if ids is None:
            # using class ID to get color
            # colors = [
            #     self._jitter([x / 255 for x in dataset_colors[c]], id) for id, c in enumerate(classes)
            # ]

            # using object ID to get color
            colors = [
                self._jitter([x / 255 for x in dataset_colors[id % len(dataset_colors)]],
                             id) for id, c in enumerate(classes)
            ]
        else:
            # using class ID to get color
            # colors = [
            #     self._jitter([x / 255 for x in dataset_colors[c]], ids[id]) for id, c in enumerate(classes)
            # ]
            # using object ID to get color
            colors = [
                self._jitter([x / 255 for x in dataset_colors[ids[id] % len(dataset_colors)]],
                             ids[id]) for id, c in enumerate(classes)
            ]
        alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.img = self._create_grayscale_image(
                (preds.pred_masks.any(dim=0) > 0).numpy()
                if preds.has("pred_masks")
                else None
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            assigned_colors=colors,
            alpha=alpha,
        )

        return self.output
