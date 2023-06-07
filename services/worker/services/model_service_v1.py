import json
import torch, torchvision
from PIL import Image
import numpy as np
import cv2
import pickle
import base64
import os
import imutils
import random

import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog

from config.config import settings
from services.modifiedvisualizer import modVisualizer

import matplotlib.image as mpimg
import ast
from matplotlib import pyplot as plt
from config.logger import logger


class DamageModel:

    predictor = None
    score_thresh = 0.20

    thing_classes = [
        "bent",
        "bonnet",
        "boot",
        "bumper dent",
        "bumper fix",
        "bumper replace",
        "crack",
        "dent major",
        "dent medium",
        "dent minor",
        "flat",
        "front bumper",
        "front grill",
        "front windshield",
        "left fender",
        "left front door",
        "left headlamp",
        "left indicator",
        "left orvm",
        "left quarter panel",
        "left rear door",
        "left running board",
        "left tail lamp",
        "number plate",
        "rear bumper",
        "rear windshield",
        "replace",
        "right fender",
        "right front door",
        "right headlamp",
        "right indicator",
        "right orvm",
        "right quarter panel",
        "right rear door",
        "right running board",
        "right tail lamp",
        "rim",
        "scratch",
        "tyre",
    ]

    damages = [
        "bumper dent",
        "bumper fix",
        "bumper replace",
        "crack",
        "dent major",
        "dent medium",
        "dent minor",
        "scratch",
        "number plate" "replace",
        "bent",
        "flat",
    ]

    parts_list = [
        1,
        2,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        38,
    ]

    dic_id = {}

    # initialize model
    def __init__(self, case_dir, update="v1_1"):
        logger.info("Using model service version 1")
        self.result_dir = f"{case_dir}/"
        self.weight_path = f"mrcnn/v1/{update}"
        self.cfg = get_cfg()
        self.cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        self.maskBlendImgDir = os.path.join(settings.project_root, "BluePrintMasks/")
        ##################################
        self.cfg.MODEL.WEIGHTS = os.path.join(
            settings.project_root, self.weight_path, "model_final.pth"
        )  ##########
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_thresh
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 39
        self.cfg.MODEL.DEVICE = "cuda"  # Cpu/gpu (1/3)
        # self.cfg.MODEL.DEVICE='cpu'
        self.predictor = DefaultPredictor(self.cfg)
        # Image size to which input images will be converted.
        self.imgSize = (800, 800)

        for i in range(len(self.thing_classes)):
            self.dic_id[i] = self.thing_classes[i]

    def image_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized

    # Input image, List of parts to be removed while generating damge mask only image
    def get_prediction(self, img):
        torch.cuda.empty_cache()
        mmimg = img.copy()
        mfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        mfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        mfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        mfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        predictor = DefaultPredictor(mfg)
        outputs = predictor(mmimg)
        # mmv = Visualizer(mmimg[:, :, ::-1], MetadataCatalog.get(mfg.DATASETS.TRAIN[0]), scale=1.2)
        # mmout = mmv.draw_instance_predictions(outputs["instances"].to("cpu"))

        # imS1 = cv2.resize(mmout.get_image()[:, :, ::-1], (540, 480))
        # plt.imshow(imS1)
        mmoutClass = outputs["instances"].pred_classes.tolist()
        mmscores = outputs["instances"].scores.tolist()
        mmnew_mask_array = outputs["instances"].pred_masks.tolist()
        mmpred_boxes = (outputs["instances"].pred_boxes).__dict__["tensor"].tolist()

        mmwrong_label_index = []
        for i in range(len(mmoutClass)):
            if mmoutClass[i] != 2:
                mmwrong_label_index.append(i)

        for i in sorted(mmwrong_label_index, reverse=True):
            # print('wrong_label_index - ',wrong_label_index[i])
            del mmoutClass[i]
            del mmscores[i]
            del mmnew_mask_array[i]
            del mmpred_boxes[i]
        # print('lenght is',len(mmoutClass))

        if len(mmoutClass) != 0:
            mmarea = []
            for i in range(len(mmoutClass)):
                mmarea.append(
                    (mmpred_boxes[i][2] - mmpred_boxes[i][0])
                    * (mmpred_boxes[i][3] - mmpred_boxes[i][1])
                )
            # print('mmarea', len(mmarea))
            # print(mmarea.index(max(mmarea)),"max is: ", max(mmarea))
            max_index = mmarea.index(max(mmarea))

            mmwrong_label_index = []
            for i in range(len(mmoutClass)):
                if i == max_index:
                    mmwrong_label_index.append(i)

            for i in sorted(mmwrong_label_index, reverse=True):
                # print('wrong_label_index - ',wrong_label_index[i])
                del mmoutClass[i]
                del mmscores[i]
                del mmnew_mask_array[i]
                del mmpred_boxes[i]
                del mmarea[i]
            # print('lenght is',len(mmoutClass))
            # print('mmarea', len(mmarea))

        mmnew_pred_classes = (torch.Tensor(mmoutClass)).int()
        mmnew_scores = torch.Tensor(mmscores)
        mmnew_pred_masks = torch.Tensor(mmnew_mask_array)
        mmnew_pred_boxes = detectron2.structures.boxes.Boxes(torch.Tensor(mmpred_boxes))

        mmobj = detectron2.structures.Instances(image_size=(480, 640))

        mmobj.set("pred_classes", (mmnew_pred_classes))
        mmobj.set("scores", (mmnew_scores))
        mmobj.set("pred_boxes", (mmnew_pred_boxes))
        mmobj.set("pred_masks", (mmnew_pred_masks))

        mmv = Visualizer(
            mmimg[:, :, ::-1], MetadataCatalog.get(mfg.DATASETS.TRAIN[0]), scale=1.2
        )
        mmout = mmv.draw_instance_predictions(mmobj.to("cpu"))
        mmoutimg = cv2.cvtColor(mmout.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
        # plt.imshow(mmoutimg)

        mmmnew_mask_array = mmobj.pred_masks.cpu().numpy()
        for i in range(len(mmmnew_mask_array)):
            mmd_img = mmmnew_mask_array[i]
            mmd_img = (mmd_img * 255).astype("uint8")
            # plt.imshow(mmd_img)

            rows = len(mmd_img)
            columns = len(mmd_img[0])

            for i in range(rows):
                for j in range(columns):
                    r = random.randint(1, 3)
                    if r == 1:
                        if mmd_img[i, j] == 255:
                            mmimg[i, j, 0] = 0
                            mmimg[i, j, 1] = 0
                            mmimg[i, j, 2] = 0
                    if r == 2:
                        if mmd_img[i, j] == 255:
                            mmimg[i, j, 0] = 0
                            mmimg[i, j, 1] = 0
                            mmimg[i, j, 2] = 0
                    if r == 3:
                        if mmd_img[i, j] == 255:
                            mmimg[i, j, 0] = 0
                            mmimg[i, j, 1] = 0
                            mmimg[i, j, 2] = 0
        # plt.imshow(mmimg)
        # print(len(mmmnew_mask_array))

        outputs = self.predictor(mmimg)
        #######################################
        f = open(
            os.path.join(
                settings.project_root,
                self.weight_path,
                "car_data_metadata.pickle",
            ),
            "rb",
        )
        car_data_metadata, cfg = pickle.load(f)
        outClass = outputs["instances"].pred_classes.tolist()
        scores = outputs["instances"].scores.tolist()
        mask_array = outputs["instances"].pred_masks.cpu().numpy()
        pred_boxes = (outputs["instances"].pred_boxes).__dict__["tensor"].tolist()
        # plt.figure(figsize = (14, 10))

        new_mask_array = outputs["instances"].pred_masks.cpu().numpy().tolist()

        label_names = []
        label_mid_point = []
        for i in range(len(outClass)):
            label_names.append(self.dic_id[outClass[i]])
            label_mid_point.append((pred_boxes[i][0] + pred_boxes[i][2]) / 2)

        """     
        print(type(outClass))
        print(type(scores))
        print(type(new_mask_array))
        print(type(pred_boxes))
        print(type(label_names))
        print(type(label_mid_point))
        
        #print((new_mask_array))
        #print(label_names)
        #print(len(label_names))
        
        """
        wrong_label_index = []

        # wrong_label_index.clear()

        for i in sorted(wrong_label_index, reverse=True):
            # print(i)
            del label_names[i]
            del outClass[i]
            del scores[i]
            del new_mask_array[i]
            del pred_boxes[i]
            del label_mid_point[i]

        # print(label_names)
        # print(len(label_names))
        wrong_label_index = []

        # to detect wrong left and right headlamp
        front = ["front bumper", "bonnet", "front grill"]
        for j in range(len(front)):
            for i in range(len(label_names)):
                if front[j] in label_names:
                    if label_names[i] == "left headlamp":
                        if (
                            label_mid_point[i]
                            < label_mid_point[label_names.index(front[j])]
                        ):
                            if "right fender" in label_names:
                                if abs(
                                    pred_boxes[i][2]
                                    - pred_boxes[label_names.index("right fender")][0]
                                ) > 2 * (pred_boxes[i][2] - pred_boxes[i][0]):
                                    # print(label_names[i], label_mid_point[i],scores[i])
                                    wrong_label_index.append(i)
                                if (
                                    label_mid_point[i]
                                    > label_mid_point[label_names.index("right fender")]
                                ):
                                    # print(label_names[i], label_mid_point[i],scores[i])
                                    wrong_label_index.append(i)
                            elif "left fender" in label_names:
                                if abs(
                                    pred_boxes[i][2]
                                    - pred_boxes[label_names.index("left fender")][0]
                                ) > 2 * (pred_boxes[i][2] - pred_boxes[i][0]):
                                    # print(label_names[i], label_mid_point[i],scores[i])
                                    wrong_label_index.append(i)
                                if (
                                    label_mid_point[i]
                                    > label_mid_point[label_names.index("left fender")]
                                ):
                                    # print(label_names[i], label_mid_point[i],scores[i])
                                    wrong_label_index.append(i)
                            else:
                                # print(label_names[i], label_mid_point[i],scores[i])
                                wrong_label_index.append(i)
                    if label_names[i] == "right headlamp":
                        if (
                            label_mid_point[i]
                            > label_mid_point[label_names.index(front[j])]
                        ):
                            if "right fender" in label_names:
                                if abs(
                                    pred_boxes[label_names.index("right fender")][2]
                                    - pred_boxes[i][0]
                                ) > 2 * (pred_boxes[i][2] - pred_boxes[i][0]):
                                    # print(label_names[i], label_mid_point[i],scores[i])
                                    wrong_label_index.append(i)
                                if (
                                    label_mid_point[i]
                                    < label_mid_point[label_names.index("right fender")]
                                ):
                                    # print(label_names[i], label_mid_point[i],scores[i])
                                    # print((label_mid_point[label_names.index('left fender')]-label_mid_point[i]),(pred_boxes[i][0]+pred_boxes[i][0]))
                                    wrong_label_index.append(i)
                            elif "left fender" in label_names:
                                if abs(
                                    pred_boxes[label_names.index("left fender")][2]
                                    - pred_boxes[i][0]
                                ) > 2 * (pred_boxes[i][2] - pred_boxes[i][0]):
                                    # print(label_names[i], label_mid_point[i],scores[i])
                                    wrong_label_index.append(i)
                                if (
                                    label_mid_point[i]
                                    < label_mid_point[label_names.index("left fender")]
                                ):
                                    # print(label_names[i], label_mid_point[i],scores[i])
                                    wrong_label_index.append(i)
                            else:
                                # print(label_names[i], label_mid_point[i],scores[i])
                                wrong_label_index.append(i)

        for i in range(len(label_names)):
            if "right fender" in label_names:
                if label_names[i] == "left headlamp":
                    if (
                        label_mid_point[i]
                        > label_mid_point[label_names.index("right fender")]
                    ):
                        if abs(
                            pred_boxes[label_names.index("right fender")][2]
                            - pred_boxes[i][0]
                        ) < (pred_boxes[i][2] - pred_boxes[i][0]):
                            # print(label_names[i], label_mid_point[i],scores[i])
                            wrong_label_index.append(i)
            if "left fender" in label_names:
                if label_names[i] == "left headlamp":
                    if (
                        label_mid_point[i]
                        > label_mid_point[label_names.index("left fender")]
                    ):
                        if abs(
                            pred_boxes[label_names.index("left fender")][2]
                            - pred_boxes[i][0]
                        ) < (pred_boxes[i][2] - pred_boxes[i][0]):
                            # print(label_names[i], label_mid_point[i],scores[i])
                            wrong_label_index.append(i)
            if "right fender" in label_names:
                if label_names[i] == "right headlamp":
                    if (
                        label_mid_point[i]
                        < label_mid_point[label_names.index("right fender")]
                    ):
                        if abs(
                            pred_boxes[label_names.index("right fender")][0]
                            - pred_boxes[i][2]
                        ) < (pred_boxes[i][2] - pred_boxes[i][0]):
                            # print(label_names[i], label_mid_point[i],scores[i])
                            wrong_label_index.append(i)
            if "left fender" in label_names:
                if label_names[i] == "right headlamp":
                    if (
                        label_mid_point[i]
                        < label_mid_point[label_names.index("left fender")]
                    ):
                        if abs(
                            pred_boxes[label_names.index("left fender")][0]
                            - pred_boxes[i][2]
                        ) < (pred_boxes[i][2] - pred_boxes[i][0]):
                            # print(label_names[i], label_mid_point[i],scores[i])
                            wrong_label_index.append(i)

        # to detect wrong left and right tail lamp
        rear = ["rear bumper", "boot", "rear windshield"]
        for j in range(len(rear)):
            for i in range(len(label_names)):
                if rear[j] in label_names:
                    if label_names[i] == "right tail lamp":
                        if (
                            label_mid_point[i]
                            < label_mid_point[label_names.index(rear[j])]
                        ):
                            if "right quarter panel" in label_names:
                                if abs(
                                    pred_boxes[i][2]
                                    - pred_boxes[
                                        label_names.index("right quarter panel")
                                    ][0]
                                ) > 2 * (pred_boxes[i][2] - pred_boxes[i][0]):
                                    # print(label_names[i], label_mid_point[i],scores[i])
                                    wrong_label_index.append(i)
                                if (
                                    label_mid_point[i]
                                    > label_mid_point[
                                        label_names.index("right quarter panel")
                                    ]
                                ):
                                    # print(label_names[i], label_mid_point[i],scores[i])
                                    wrong_label_index.append(i)
                            elif "left quarter panel" in label_names:
                                if (
                                    label_mid_point[i]
                                    > label_mid_point[
                                        label_names.index("left quarter panel")
                                    ]
                                ):
                                    # print(label_names[i], label_mid_point[i],scores[i])
                                    wrong_label_index.append(i)
                                if abs(
                                    pred_boxes[i][2]
                                    - pred_boxes[
                                        label_names.index("left quarter panel")
                                    ][0]
                                ) > 2 * (pred_boxes[i][2] - pred_boxes[i][0]):
                                    # print(label_names[i], label_mid_point[i],scores[i])
                                    wrong_label_index.append(i)
                            else:
                                # print(label_names[i], label_mid_point[i],scores[i])
                                wrong_label_index.append(i)
                    if label_names[i] == "left tail lamp":
                        if (
                            label_mid_point[i]
                            > label_mid_point[label_names.index(rear[j])]
                        ):
                            if "right quarter panel" in label_names:
                                if abs(
                                    pred_boxes[
                                        label_names.index("right quarter panel")
                                    ][2]
                                    - pred_boxes[i][0]
                                ) > 2 * (pred_boxes[i][2] - pred_boxes[i][0]):
                                    # print(label_names[i], label_mid_point[i],scores[i])
                                    wrong_label_index.append(i)
                                if (
                                    label_mid_point[i]
                                    < label_mid_point[
                                        label_names.index("right quarter panel")
                                    ]
                                ):
                                    # print(label_names[i], label_mid_point[i],scores[i])
                                    wrong_label_index.append(i)
                            elif "left quarter panel" in label_names:
                                # print(label_names[i], label_mid_point[i],scores[i])
                                if (
                                    label_mid_point[i]
                                    < label_mid_point[
                                        label_names.index("left quarter panel")
                                    ]
                                ):
                                    # print(label_names[i], label_mid_point[i],scores[i])
                                    wrong_label_index.append(i)
                                if abs(
                                    pred_boxes[label_names.index("left quarter panel")][
                                        2
                                    ]
                                    - pred_boxes[i][0]
                                ) > 2 * (pred_boxes[i][2] - pred_boxes[i][0]):
                                    # print(label_names[i], label_mid_point[i],scores[i])
                                    wrong_label_index.append(i)
                            else:
                                # print(label_names[i], label_mid_point[i],scores[i])
                                wrong_label_index.append(i)

        for i in range(len(label_names)):
            if "right quarter panel" in label_names:
                if label_names[i] == "left tail lamp":
                    if (
                        label_mid_point[i]
                        < label_mid_point[label_names.index("right quarter panel")]
                    ):
                        # print(label_names[i], label_mid_point[i],scores[i])
                        if abs(
                            pred_boxes[label_names.index("right quarter panel")][0]
                            - pred_boxes[i][2]
                        ) < (pred_boxes[i][2] - pred_boxes[i][0]):
                            # print(label_names[i], label_mid_point[i],scores[i])
                            wrong_label_index.append(i)
            if "left quarter panel" in label_names:
                if label_names[i] == "left tail lamp":
                    if (
                        label_mid_point[i]
                        < label_mid_point[label_names.index("left quarter panel")]
                    ):
                        # print(label_names[i], label_mid_point[i],scores[i])
                        if abs(
                            pred_boxes[label_names.index("left quarter panel")][0]
                            - pred_boxes[i][2]
                        ) < (pred_boxes[i][2] - pred_boxes[i][0]):
                            # print(label_names[i], label_mid_point[i],scores[i])
                            wrong_label_index.append(i)
            if "right quarter panel" in label_names:
                if label_names[i] == "right tail lamp":
                    if (
                        label_mid_point[i]
                        > label_mid_point[label_names.index("right quarter panel")]
                    ):
                        if abs(
                            pred_boxes[label_names.index("right quarter panel")][2]
                            - pred_boxes[i][0]
                        ) < (pred_boxes[i][2] - pred_boxes[i][0]):
                            # print(label_names[i], label_mid_point[i],scores[i])
                            wrong_label_index.append(i)
            if "left quarter panel" in label_names:
                if label_names[i] == "right tail lamp":
                    if (
                        label_mid_point[i]
                        > label_mid_point[label_names.index("left quarter panel")]
                    ):
                        if abs(
                            pred_boxes[label_names.index("left quarter panel")][2]
                            - pred_boxes[i][0]
                        ) < (pred_boxes[i][2] - pred_boxes[i][0]):
                            # print(label_names[i], label_mid_point[i],scores[i])
                            wrong_label_index.append(i)

        # to detected wrong side using front and rear door
        for i in range(len(label_names)):
            if "left front door" in label_names and "left rear door" in label_names:
                if (
                    label_mid_point[label_names.index("left rear door")]
                    > label_mid_point[label_names.index("left front door")]
                ):
                    # print('This is left side')
                    # print(label_names.index('right front door'),label_names.index('right rear door'))
                    if "right front door" in label_names:
                        if label_names[i] == "right front door":
                            wrong_label_index.append(i)
                    if "right rear door" in label_names:
                        if label_names[i] == "right rear door":
                            wrong_label_index.append(i)
                    if "right fender" in label_names:
                        if label_names[i] == "right fender":
                            wrong_label_index.append(i)
                    if "right quarter panel" in label_names:
                        if label_names[i] == "right quarter panel":
                            wrong_label_index.append(i)
                    if "right running board" in label_names:
                        if label_names[i] == "right running board":
                            wrong_label_index.append(i)
                    if "right indicator" in label_names:
                        if label_names[i] == "right indicator":
                            wrong_label_index.append(i)
                    if "right orvm" in label_names:
                        if label_names[i] == "right orvm":
                            wrong_label_index.append(i)

        for i in range(len(label_names)):
            if "right front door" in label_names and "right rear door" in label_names:
                if (
                    label_mid_point[label_names.index("right rear door")]
                    < label_mid_point[label_names.index("right front door")]
                ):
                    # print('This is right side')
                    # print(label_names.index('left front door'),label_names.index('left rear door'))
                    if "left front door" in label_names:
                        if label_names[i] == "left front door":
                            wrong_label_index.append(i)
                    if "left rear door" in label_names:
                        if label_names[i] == "left rear door":
                            wrong_label_index.append(i)
                    if "left fender" in label_names:
                        if label_names[i] == "left fender":
                            wrong_label_index.append(i)
                    if "left quarter panel" in label_names:
                        if label_names[i] == "left quarter panel":
                            wrong_label_index.append(i)
                    if "left running board" in label_names:
                        if label_names[i] == "left running board":
                            wrong_label_index.append(i)
                    if "left indicator" in label_names:
                        if label_names[i] == "left indicator":
                            wrong_label_index.append(i)
                    if "left orvm" in label_names:
                        if label_names[i] == "left orvm":
                            wrong_label_index.append(i)

        # to detected wrong side using front door and fender
        for i in range(len(label_names)):
            if "left front door" in label_names and "left fender" in label_names:
                if (
                    label_mid_point[label_names.index("left front door")]
                    > label_mid_point[label_names.index("left fender")]
                ):
                    # print('This is left side')
                    # print(label_names.index('right front door'),label_names.index('right rear door'))
                    if "right front door" in label_names:
                        if label_names[i] == "right front door":
                            wrong_label_index.append(i)
                    if "right fender" in label_names:
                        if label_names[i] == "right fender":
                            wrong_label_index.append(i)
                    if "right rear door" in label_names:
                        if label_names[i] == "right rear door":
                            wrong_label_index.append(i)
                    if "right quarter panel" in label_names:
                        if label_names[i] == "right quarter panel":
                            wrong_label_index.append(i)
                    if "right running board" in label_names:
                        if label_names[i] == "right running board":
                            wrong_label_index.append(i)
                    if "right indicator" in label_names:
                        if label_names[i] == "right indicator":
                            wrong_label_index.append(i)
                    if "right orvm" in label_names:
                        if label_names[i] == "right orvm":
                            wrong_label_index.append(i)

        for i in range(len(label_names)):
            if "right front door" in label_names and "right fender" in label_names:
                if (
                    label_mid_point[label_names.index("right front door")]
                    < label_mid_point[label_names.index("right fender")]
                ):
                    # print('This is right side')
                    # print(label_names.index('left front door'),label_names.index('left rear door'))
                    if "left front door" in label_names:
                        if label_names[i] == "left front door":
                            wrong_label_index.append(i)
                    if "left fender" in label_names:
                        if label_names[i] == "left fender":
                            wrong_label_index.append(i)
                    if "left rear door" in label_names:
                        if label_names[i] == "left rear door":
                            wrong_label_index.append(i)
                    if "left quarter panel" in label_names:
                        if label_names[i] == "left quarter panel":
                            wrong_label_index.append(i)
                    if "left running board" in label_names:
                        if label_names[i] == "left running board":
                            wrong_label_index.append(i)
                    if "left indicator" in label_names:
                        if label_names[i] == "left indicator":
                            wrong_label_index.append(i)
                    if "left orvm" in label_names:
                        if label_names[i] == "left orvm":
                            wrong_label_index.append(i)

        # to detected wrong side using door and quarter panel
        for i in range(len(label_names)):
            if "left rear door" in label_names and "left quarter panel" in label_names:
                if (
                    label_mid_point[label_names.index("left quarter panel")]
                    > label_mid_point[label_names.index("left rear door")]
                ):
                    # print('This is left side')
                    # print(label_names.index('right front door'),label_names.index('right rear door'))
                    if "right front door" in label_names:
                        if label_names[i] == "right front door":
                            wrong_label_index.append(i)
                    if "right rear door" in label_names:
                        if label_names[i] == "right rear door":
                            wrong_label_index.append(i)
                    if "right fender" in label_names:
                        if label_names[i] == "right fender":
                            wrong_label_index.append(i)
                    if "right quarter panel" in label_names:
                        if label_names[i] == "right quarter panel":
                            wrong_label_index.append(i)
                    if "right running board" in label_names:
                        if label_names[i] == "right running board":
                            wrong_label_index.append(i)
                    if "right indicator" in label_names:
                        if label_names[i] == "right indicator":
                            wrong_label_index.append(i)
                    if "right orvm" in label_names:
                        if label_names[i] == "right orvm":
                            wrong_label_index.append(i)

        for i in range(len(label_names)):
            if (
                "right rear door" in label_names
                and "right quarter panel" in label_names
            ):
                if (
                    label_mid_point[label_names.index("right quarter panel")]
                    < label_mid_point[label_names.index("right rear door")]
                ):
                    # print('This is right side')
                    # print(label_names.index('left front door'),label_names.index('left rear door'))
                    if "left front door" in label_names:
                        if label_names[i] == "left front door":
                            wrong_label_index.append(i)
                    if "left rear door" in label_names:
                        if label_names[i] == "left rear door":
                            wrong_label_index.append(i)
                    if "left fender" in label_names:
                        if label_names[i] == "left fender":
                            wrong_label_index.append(i)
                    if "left quarter panel" in label_names:
                        if label_names[i] == "left quarter panel":
                            wrong_label_index.append(i)
                    if "left running board" in label_names:
                        if label_names[i] == "left running board":
                            wrong_label_index.append(i)
                    if "left indicator" in label_names:
                        if label_names[i] == "left indicator":
                            wrong_label_index.append(i)
                    if "left orvm" in label_names:
                        if label_names[i] == "left orvm":
                            wrong_label_index.append(i)

        # to detect wrong left and right orvm
        for i in range(len(label_names)):
            if (
                "front windshield" in label_names
                and "left front door" not in label_names
                and "right front door" not in label_names
                and "left rear door" not in label_names
                and "right rear door" not in label_names
                and "left quarter panel" not in label_names
                and "right quarter panel" not in label_names
            ):
                if label_names[i] == "left orvm":
                    if (
                        label_mid_point[i]
                        < label_mid_point[label_names.index("front windshield")]
                    ):
                        # print(label_names[i], label_mid_point[i],scores[i])
                        wrong_label_index.append(i)
                if label_names[i] == "right orvm":
                    if (
                        label_mid_point[i]
                        > label_mid_point[label_names.index("front windshield")]
                    ):
                        # print(label_names[i], label_mid_point[i],scores[i])
                        wrong_label_index.append(i)

        # to detect wrong left and right fender
        for i in range(len(label_names)):
            if (
                "left headlamp" in label_names
                or "right headlamp" in label_names
                and "left front door" not in label_names
                and "right front door" not in label_names
                and "left rear door" not in label_names
                and "right rear door" not in label_names
                and "left quarter panel" not in label_names
                and "right quarter panel" not in label_names
            ):
                if label_names[i] == "left fender":
                    if "left headlamp" in label_names:
                        if (
                            label_mid_point[i]
                            < label_mid_point[label_names.index("left headlamp")]
                        ):
                            # print(label_names[i], label_mid_point[i],scores[i])
                            wrong_label_index.append(i)
                if label_names[i] == "left fender":
                    if "right headlamp" in label_names:
                        if (
                            label_mid_point[i]
                            < label_mid_point[label_names.index("right headlamp")]
                        ):
                            # print(label_names[i], label_mid_point[i],scores[i])
                            wrong_label_index.append(i)
                if label_names[i] == "right fender":
                    if "left headlamp" in label_names:
                        if (
                            label_mid_point[i]
                            > label_mid_point[label_names.index("left headlamp")]
                        ):
                            # print(label_names[i], label_mid_point[i],scores[i])
                            wrong_label_index.append(i)
                if label_names[i] == "right fender":
                    if "right headlamp" in label_names:
                        if (
                            label_mid_point[i]
                            > label_mid_point[label_names.index("right headlamp")]
                        ):
                            # print(label_names[i], label_mid_point[i],scores[i])
                            wrong_label_index.append(i)

        # to detect wrong left and right quarter panel
        for i in range(len(label_names)):
            if (
                "left tail lamp" in label_names
                or "right tail lamp" in label_names
                and "left front door" not in label_names
                and "right front door" not in label_names
                and "left rear door" not in label_names
                and "right rear door" not in label_names
                and "left fender" not in label_names
                and "right fender" not in label_names
            ):
                if label_names[i] == "left quarter panel":
                    if "left tail lamp" in label_names:
                        if (
                            label_mid_point[i]
                            > label_mid_point[label_names.index("left tail lamp")]
                        ):
                            # print(label_names[i], label_mid_point[i],scores[i])
                            wrong_label_index.append(i)
                if label_names[i] == "left quarter panel":
                    if "right tail lamp" in label_names:
                        if (
                            label_mid_point[i]
                            > label_mid_point[label_names.index("right tail lamp")]
                        ):
                            # print(label_names[i], label_mid_point[i],scores[i])
                            wrong_label_index.append(i)
                if "right quarter panel" in label_names:
                    if label_names[i] == "left tail lamp":
                        if (
                            label_mid_point[i]
                            < label_mid_point[label_names.index("left tail lamp")]
                        ):
                            # print(label_names[i], label_mid_point[i],scores[i])
                            wrong_label_index.append(i)
                if label_names[i] == "right quarter panel":
                    if "right tail lamp" in label_names:
                        if (
                            label_mid_point[i]
                            < label_mid_point[label_names.index("right tail lamp")]
                        ):
                            # print(label_names[i], label_mid_point[i],scores[i])
                            wrong_label_index.append(i)

        # print(wrong_label_index)
        # print(label_names)
        # for i in range(len(wrong_label_index)):
        # print(label_names[wrong_label_index[i]])
        wrong_label_index = set(wrong_label_index)

        # wrong_label_index.clear()

        for i in sorted(wrong_label_index, reverse=True):
            # print(i)
            del label_names[i]
            del outClass[i]
            del scores[i]
            del new_mask_array[i]
            del pred_boxes[i]
            del label_mid_point[i]

        wrong_label_index.clear()
        wrong_label_index = []

        # to detect and remove same name panel name with low accuracy
        temp_panel_names = [
            "bonnet",
            "boot",
            "front bumper",
            "front grill",
            "front windshield",
            "left fender",
            "left front door",
            "left headlamp",
            "left indicator",
            "left orvm",
            "left quarter panel",
            "left rear door",
            "left running board",
            "left tail lamp",
            "number plate",
            "rear bumper",
            "rear windshield",
            "right fender",
            "right front door",
            "right headlamp",
            "right indicator",
            "right orvm",
            "right quarter panel",
            "right rear door",
            "right running board",
            "right tail lamp",
        ]
        for j in range(len(temp_panel_names)):
            if temp_panel_names[j] in label_names:
                cnt = 0
                for i in range(len(label_names)):
                    if label_names[i] == temp_panel_names[j]:
                        cnt += 1
                if cnt > 1:
                    temp_index = []
                    for i in range(len(label_names)):
                        if label_names[i] == temp_panel_names[j]:
                            temp_index.append(i)
                    temp_index = sorted(temp_index[1:])
                    for i in range(len(temp_index)):
                        wrong_label_index.append(temp_index[i])
        # wrong_label_index.clear()
        wrong_label_index = set(wrong_label_index)
        for i in sorted(wrong_label_index, reverse=True):
            # print(i)
            del label_names[i]
            del outClass[i]
            del scores[i]
            del new_mask_array[i]
            del pred_boxes[i]
            del label_mid_point[i]

        # print(label_names)
        # print(len(label_names))

        new_pred_classes = (torch.Tensor(outClass)).int()
        new_scores = torch.Tensor(scores)
        new_pred_masks = torch.Tensor(new_mask_array)
        new_pred_boxes = detectron2.structures.boxes.Boxes(torch.Tensor(pred_boxes))
        # print(new_mask_array)

        # print((new_pred_classes))
        # print((new_scores))
        # print((new_pred_boxes))
        # print((new_pred_masks))

        obj = detectron2.structures.Instances(image_size=(480, 640))

        obj.set("pred_classes", (new_pred_classes))
        obj.set("scores", (new_scores))
        obj.set("pred_boxes", (new_pred_boxes))
        obj.set("pred_masks", (new_pred_masks))

        outClass = obj.pred_classes.tolist()
        scores = obj.scores.tolist()
        mask_array = obj.pred_masks.cpu().numpy()
        pred_boxes = (obj.pred_boxes).__dict__["tensor"].tolist()

        v = Visualizer(
            img[:, :, ::-1],
            metadata=car_data_metadata,
            scale=1,
            # instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        out = v.draw_instance_predictions(obj.to("cpu"))
        outimg = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
        # plt.imshow(outimg)

        # Make damge mask only image(Filter part masks)
        mV = modVisualizer(
            img[:, :, ::-1],
            metadata=car_data_metadata,
            scale=1,
            instance_mode=ColorMode.IMAGE_BW,  # remove the colors of unsegmented pixels
        )

        mV = mV.draw_instance_predictions(
            outputs["instances"].to("cpu"), self.parts_list
        )
        dmgMaskImg = cv2.cvtColor(mV.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)

        # Accident type determination

        damageTypeOfCar = []

        majorDamages = [24, 6, 4]  # Replace, Dent major, Bumper replace
        mediumDamages = [3, 7, 5]  # Bumper fix, Dent medium, crack
        minorDamages = [8, 2, 34]  # Dent minor, Bumper dent, Scratch

        numMajorDamages = 0
        numMediumDamages = 0
        numMinorDamages = 0

        # Occurance of major damage calculate
        numMajorDamages = sum(
            [outClass.count(majorDamages[i]) for i in range(len(majorDamages))]
        )
        # print('Number of major damages: %d' %numMajorDamages)

        # Occurance of medium damage calculate
        numMediumDamages = sum(
            [outClass.count(mediumDamages[i]) for i in range(len(mediumDamages))]
        )
        # print('Number of medium damages: %d' %numMediumDamages)

        # Occurance of minor damage calculate
        numMinorDamages = sum(
            [outClass.count(minorDamages[i]) for i in range(len(minorDamages))]
        )
        # print('Number of minor damages: %d' %numMinorDamages)

        # Set some threshold for the determination of damge type
        majorDamageThresh = 1
        mediumDamageThresh = 1
        minorDamageThresh = 1

        # Check the number of damages and do the damge type determination.
        if numMajorDamages >= majorDamageThresh:
            damageTypeOfCar = "MAJOR"

        elif numMediumDamages >= mediumDamageThresh:
            damageTypeOfCar = "MEDIUM"

        elif numMinorDamages >= minorDamageThresh:
            damageTypeOfCar = "MINOR"

        # Damage location determination(bbox intersection)

        panel_label = []
        damage_label = []
        panel_box = []
        damage_box = []
        panel_score = []
        damage_score = []
        for i in range(len(outClass)):
            if self.dic_id[outClass[i]] == "number plate":
                panel_label.append(self.dic_id[outClass[i]])
                panel_box.append(pred_boxes[i])
                panel_score.append(scores[i])
            elif self.dic_id[outClass[i]] in self.damages:
                damage_label.append(self.dic_id[outClass[i]])
                damage_box.append(pred_boxes[i])
                damage_score.append(scores[i])
            else:
                panel_label.append(self.dic_id[outClass[i]])
                panel_box.append(pred_boxes[i])
                panel_score.append(scores[i])

        identified_damages = []  # for damages which are only identified on the panel.

        final_dict = {}
        final_damage_dict = {}
        for i in range(len(panel_box)):
            pbox = panel_box[i]
            damage_arr = []
            damage_list = []
            boundb = {"x1": pbox[0], "y1": pbox[1], "x4": pbox[2], "y4": pbox[3]}
            for j in range(len(damage_box)):
                dbox = damage_box[j]
                # Inner box
                innerb = {"x1": dbox[0], "y1": dbox[1], "x4": dbox[2], "y4": dbox[3]}
                # If top-left inner box corner is inside the bounding box
                if boundb["x1"] < innerb["x1"] and boundb["y1"] < innerb["y1"]:
                    # If bottom-right inner box corner is inside the bounding box
                    if boundb["x4"] > innerb["x4"] and boundb["y4"] > innerb["y4"]:
                        # damage_arr.append(damage_label[j])
                        damage_list.append(damage_label[j])
                        damage_arr.append(
                            {damage_label[j]: "{:.2f}".format(damage_score[j] * 100)}
                        )

            # if(((boundb['x1'] + boundb['x4'])/2) not in wrong_label_loc):
            identified_damages.extend(
                damage_arr
            )  # Collecting all identified damages from panel.

            if len(damage_list) > 0:
                final_dict[panel_label[i]] = damage_arr
                final_damage_dict[panel_label[i]] = list(set(damage_list))
            else:
                if (
                    panel_label[i] not in final_dict.keys()
                ):  # if panal is not present in final_dict then only add this.
                    final_dict[panel_label[i]] = []
                    final_damage_dict[panel_label[i]] = []

        # Collecting all damages which are identified and unidentified
        all_damages = []
        for i in range(len(damage_label)):
            all_damages.append(
                {damage_label[i]: "{:.2f}".format(damage_score[i] * 100)}
            )

        # Separating unidentified damage on panel from all damages
        unidentified_damages = []
        for i in all_damages:
            if (
                len(identified_damages) == 0
            ):  # if there is not a single identified damage then all damage as unindetified.
                unidentified_damages = all_damages
                break
            for j in identified_damages:
                if i != j:
                    unidentified_damages.append(i)

        # Removeing duplicate damage from unidentified list
        seen = set()
        new_unidentified_damages = []
        for d in unidentified_damages:
            t = tuple(d.items())
            if t not in seen:
                seen.add(t)
                new_unidentified_damages.append(d)

        # Removeing duplicate damage from Unidentified Panel if it is detected by panal also.
        for k, v in final_dict.items():
            if len(v) != 0:
                for j in v:
                    for i in new_unidentified_damages:
                        if i == j:
                            new_unidentified_damages.remove(i)

        # Creating new dictonary for Unidentified Panel and damage
        unidentified_dict = {}
        unidentified_dict["Unidentified Panel"] = new_unidentified_damages

        # Mergeing two dict
        def merge_dictionary(x, y):
            """Given two dicts, merge them into a new dict as a shallow copy."""
            z = x.copy()
            z.update(y)
            return z

        # Mergeing two dictionary final_dict and unidentified_dict
        final_dict = merge_dictionary(final_dict, unidentified_dict)

        # Adding Unidentified Panel in final_damage_dict.
        final_damage_dict["Unidentified Panel"] = []
        for i in range(len(new_unidentified_damages)):
            for key, value in (new_unidentified_damages[i]).items():
                final_damage_dict["Unidentified Panel"].append(key)

        # 1st Top view image create from here.
        TPview = cv2.cvtColor(
            mpimg.imread(
                os.path.join(
                    self.maskBlendImgDir,
                    "carAllViewsbase.jpg",
                )
            ),
            cv2.COLOR_BGR2RGB,
        )

        tpview_score = scores.copy()
        for index, item in enumerate(tpview_score):
            tpview_score[index] = "{:.2f}".format(tpview_score[index] * 100)

        def mapping(x, y, y1, x1, y4, x4):
            if (
                panel_box[x][0] <= damage_box[y][0]
                and panel_box[x][1] <= damage_box[y][1]
                and panel_box[x][2] >= damage_box[y][2]
                and panel_box[x][3] >= damage_box[y][3]
            ):
                index = tpview_score.index(damage_score[y])
                d_img = mask_array[index]
                d_img = (d_img * 255).astype("uint8")
                d_img = d_img[
                    int(damage_box[y][1]) : int(damage_box[y][3]),
                    int(damage_box[y][0]) : int(damage_box[y][2]),
                ]

                panel_height = panel_box[x][3] - panel_box[x][1]
                panel_width = panel_box[x][2] - panel_box[x][0]
                damage_height = damage_box[y][3] - damage_box[y][1]
                damage_width = damage_box[y][2] - damage_box[y][0]

                h_ratio = panel_height / ((damage_box[y][1] - panel_box[x][1]))
                w_ratio = panel_width / ((damage_box[y][0] - panel_box[x][0]))

                if h_ratio >= 0 and w_ratio >= 0:
                    dy1 = ((y4 - y1) / h_ratio) + y1
                    dx1 = ((x4 - x1) / w_ratio) + x1

                    dy4 = ((y4 - y1) / (panel_height / damage_height)) + dy1
                    dx4 = ((x4 - x1) / (panel_width / damage_width)) + dx1

                    dx1 = int(dx1)
                    dx4 = int(dx4)
                    dy1 = int(dy1)
                    dy4 = int(dy4)

                    # print(damage_label[y],damage_score[y])
                    # print("Damage Location 1nd: ",dx1,dy1,dx4,dy4)

                    dx1 = abs(dx1)
                    dx4 = abs(dx4)
                    dy1 = abs(dy1)
                    dy4 = abs(dy4)
                    # print("---------1",dx4-dx1, dy4-dy1)
                    w = dx4 - dx1
                    h = dy4 - dy1

                    if w == 0:
                        w = 1
                    if h == 0:
                        h = 1
                    d_img = cv2.resize(d_img, (w, h), interpolation=cv2.INTER_NEAREST)

                    d_img_lst = []
                    rows = len(d_img)
                    columns = len(d_img[0])
                    cnt = 0
                    for i in range(rows):
                        for j in range(columns):
                            cnt += 1
                            if d_img[i, j] == 255:
                                d_img_lst.append(cnt)

                    d_lst = []
                    cnt2 = 0
                    rows = dy4 - dy1
                    columns = dx4 - dx1

                    for i in range(rows):
                        for j in range(columns):
                            cnt2 += 1
                            if cnt2 in d_img_lst:
                                d_lst.append(cnt2)
                                TPview[(i + dy1), (j + dx1), 0] = 255
                                TPview[(i + dy1), (j + dx1), 1] = 0
                                TPview[(i + dy1), (j + dx1), 2] = 0

        for index, item in enumerate(damage_score):
            damage_score[index] = "{:.2f}".format(damage_score[index] * 100)

        for k, v in final_dict.items():
            if len(v) != 0 and k != "Unidentified Panel":
                for i in v:
                    for a, b in i.items():
                        all_indexes = []
                        for i in range(0, len(panel_label)):
                            if panel_label[i] == k:
                                all_indexes.append(i)
                        for i in all_indexes:
                            index = i
                            index2 = damage_score.index(b)
                            # print(panel_label[index],damage_label[index2],index,index2)
                            if panel_label[index] == "boot":
                                mapping(index, index2, 1158, 107, 1261, 458)
                                TPview = cv2.rotate(TPview, cv2.ROTATE_90_CLOCKWISE)
                                mapping(index, index2, 1130, 1052, 1197, 1340)
                                TPview = cv2.rotate(
                                    TPview, cv2.ROTATE_90_COUNTERCLOCKWISE
                                )

                            if panel_label[index] == "left front door":
                                mapping(index, index2, 781, 842, 920, 1090)
                                mapping(index, index2, 528, 585, 543, 773)

                            if panel_label[index] == "left rear door":
                                mapping(index, index2, 770, 1089, 910, 1308)
                                mapping(index, index2, 525, 779, 539, 995)

                            if panel_label[index] == "front bumper":
                                mapping(index, index2, 863, 94, 944, 467)
                                mapping(index, index2, 856, 581, 942, 656)
                                mapping(index, index2, 1273, 1464, 1348, 1531)
                                TPview = cv2.rotate(
                                    TPview, cv2.ROTATE_90_COUNTERCLOCKWISE
                                )
                                mapping(index, index2, 1295, 231, 1328, 483)
                                TPview = cv2.rotate(TPview, cv2.ROTATE_90_CLOCKWISE)

                            if panel_label[index] == "left fender":
                                mapping(index, index2, 782, 680, 852, 845)
                                mapping(index, index2, 518, 414, 544, 546)

                            if panel_label[index] == "right fender":
                                mapping(index, index2, 1181, 1272, 1245, 1437)
                                mapping(index, index2, 172, 412, 197, 543)

                            if panel_label[index] == "left quarter panel":
                                mapping(index, index2, 702, 1297, 829, 1443)
                                mapping(index, index2, 485, 1006, 543, 1152)

                            if panel_label[index] == "rear bumper":
                                mapping(index, index2, 1257, 90, 1334, 462)
                                mapping(index, index2, 848, 1455, 931, 1525)
                                mapping(index, index2, 1245, 578, 1338, 664)
                                TPview = cv2.rotate(TPview, cv2.ROTATE_90_CLOCKWISE)
                                mapping(index, index2, 1204, 1064, 1226, 1345)
                                TPview = cv2.rotate(
                                    TPview, cv2.ROTATE_90_COUNTERCLOCKWISE
                                )

                            if panel_label[index] == "rear windshield":
                                TPview = cv2.rotate(TPview, cv2.ROTATE_90_CLOCKWISE)
                                mapping(index, index2, 1000, 1078, 1126, 1323)
                                TPview = cv2.rotate(
                                    TPview, cv2.ROTATE_90_COUNTERCLOCKWISE
                                )

                            if panel_label[index] == "front windshield":
                                TPview = cv2.rotate(
                                    TPview, cv2.ROTATE_90_COUNTERCLOCKWISE
                                )
                                mapping(index, index2, 935, 240, 1090, 481)
                                TPview = cv2.rotate(TPview, cv2.ROTATE_90_CLOCKWISE)

                            if panel_label[index] == "bonnet":
                                TPview = cv2.rotate(
                                    TPview, cv2.ROTATE_90_COUNTERCLOCKWISE
                                )
                                mapping(index, index2, 1116, 252, 1311, 452)
                                TPview = cv2.rotate(TPview, cv2.ROTATE_90_CLOCKWISE)
                                mapping(index, index2, 773, 178, 827, 378)

                            if panel_label[index] == "left tail lamp":
                                mapping(index, index2, 1178, 111, 1228, 185)
                                mapping(index, index2, 788, 1434, 832, 1504)
                                mapping(index, index2, 515, 1159, 526, 1196)

                            if panel_label[index] == "right tail lamp":
                                mapping(index, index2, 1187, 602, 1228, 678)
                                mapping(index, index2, 1181, 377, 1227, 455)
                                mapping(index, index2, 184, 1145, 205, 1197)

                            if panel_label[index] == "right headlamp":
                                mapping(index, index2, 820, 107, 860, 183)
                                mapping(index, index2, 1223, 1435, 1254, 1510)
                                TPview = cv2.rotate(
                                    TPview, cv2.ROTATE_90_COUNTERCLOCKWISE
                                )
                                mapping(index, index2, 1234, 194, 1264, 234)
                                TPview = cv2.rotate(TPview, cv2.ROTATE_90_CLOCKWISE)

                            if panel_label[index] == "left headlamp":
                                mapping(index, index2, 819, 613, 852, 672)
                                mapping(index, index2, 822, 388, 855, 449)
                                TPview = cv2.rotate(
                                    TPview, cv2.ROTATE_90_COUNTERCLOCKWISE
                                )
                                mapping(index, index2, 1246, 482, 1260, 522)
                                TPview = cv2.rotate(TPview, cv2.ROTATE_90_CLOCKWISE)

                            if panel_label[index] == "right front door":
                                mapping(index, index2, 1175, 1027, 1320, 1240)
                                mapping(index, index2, 177, 590, 188, 772)

                            if panel_label[index] == "right rear door":
                                mapping(index, index2, 1164, 822, 1322, 1016)
                                mapping(index, index2, 175, 775, 188, 977)

                            if panel_label[index] == "right quarter panel":
                                mapping(index, index2, 1111, 686, 1230, 830)
                                mapping(index, index2, 171, 1005, 220, 1143)

                            if panel_label[index] == "right orvm":
                                mapping(index, index2, 1148, 456, 1171, 495)
                                mapping(index, index2, 752, 66, 777, 109)
                                mapping(index, index2, 1153, 1188, 1177, 1236)
                                mapping(index, index2, 147, 585, 176, 614)

                            if panel_label[index] == "left orvm":
                                mapping(index, index2, 752, 887, 777, 925)
                                mapping(index, index2, 749, 453, 777, 498)
                                mapping(index, index2, 544, 580, 574, 620)
                                mapping(index, index2, 1147, 65, 1177, 106)

        # Aspect ratio preserved resize
        TPview = self.image_resize(TPview, height=800)

        # 2nd Top view image for showing two vehicle possible accident.

        front_parts = {}
        rear_parts = {}
        left_parts = {}
        right_parts = {}

        # Finding which side of car through number of panel names.
        for k, v in final_dict.items():
            if len(v) != 0 and k != "Unidentified Panel":
                if (
                    k == "bonnet"
                    or k == "front bumper"
                    or k == "front grill"
                    or k == "front windshield"
                    or k == "left headlamp"
                    or k == "right headlamp"
                ):
                    front_parts[k] = v

                if (
                    k == "left fender"
                    or k == "left front door"
                    or k == "left indicator"
                    or k == "left orvm"
                    or k == "left quarter panel"
                    or k == "left rear door"
                    or k == "left running board"
                ):
                    left_parts[k] = v

                if (
                    k == "right fender"
                    or k == "right front door"
                    or k == "right indicator"
                    or k == "right orvm"
                    or k == "right quarter panel"
                    or k == "right rear door"
                    or k == "right running board"
                ):
                    right_parts[k] = v

                if (
                    k == "boot"
                    or k == "rear bumper"
                    or k == "rear windshield"
                    or k == "left tail lamp"
                    or k == "right tail lamp"
                ):
                    rear_parts[k] = v

        def mapping2(x, y, y1, x1, y4, x4):
            if (
                panel_box[x][0] <= damage_box[y][0]
                and panel_box[x][1] <= damage_box[y][1]
                and panel_box[x][2] >= damage_box[y][2]
                and panel_box[x][3] >= damage_box[y][3]
            ):
                index = tpview_score.index(damage_score[y])
                d_img = mask_array[index]
                d_img = (d_img * 255).astype("uint8")
                d_img = d_img[
                    int(damage_box[y][1]) : int(damage_box[y][3]),
                    int(damage_box[y][0]) : int(damage_box[y][2]),
                ]
                # _ = plt.imshow(d_img)
                panel_height = panel_box[x][3] - panel_box[x][1]
                panel_width = panel_box[x][2] - panel_box[x][0]
                damage_height = damage_box[y][3] - damage_box[y][1]
                damage_width = damage_box[y][2] - damage_box[y][0]

                h_ratio = panel_height / ((damage_box[y][1] - panel_box[x][1]))
                w_ratio = panel_width / ((damage_box[y][0] - panel_box[x][0]))

                if h_ratio >= 0 and w_ratio >= 0:
                    # print("ratio", h_ratio, w_ratio)
                    dy1 = ((y4 - y1) / h_ratio) + y1
                    dx1 = ((x4 - x1) / w_ratio) + x1

                    dy4 = ((y4 - y1) / (panel_height / damage_height)) + dy1
                    dx4 = ((x4 - x1) / (panel_width / damage_width)) + dx1

                    # print("Damage Location 1nd: ",dx1,dy1,dx4,dy4)
                    dx1 = int(dx1)
                    dx4 = int(dx4)
                    dy1 = int(dy1)
                    dy4 = int(dy4)

                    dx1 = abs(dx1)
                    dx4 = abs(dx4)
                    dy1 = abs(dy1)
                    dy4 = abs(dy4)
                    # print(damage_label[y],damage_score[y])
                    # print("---------",dx4-dx1, dy4-dy1)
                    w = dx4 - dx1
                    h = dy4 - dy1
                    if w == 0:
                        w = 1
                    if h == 0:
                        h = 1
                    d_img = cv2.resize(d_img, (w, h), interpolation=cv2.INTER_NEAREST)

                    d_img_lst = []
                    rows = len(d_img)
                    columns = len(d_img[0])
                    cnt = 0
                    for i in range(rows):
                        for j in range(columns):
                            cnt += 1
                            if d_img[i, j] == 255:
                                d_img_lst.append(cnt)

                    d_lst = []
                    cnt2 = 0
                    rows = dy4 - dy1
                    columns = dx4 - dx1

                    for i in range(rows):
                        for j in range(columns):
                            cnt2 += 1
                            if cnt2 in d_img_lst:
                                d_lst.append(cnt2)
                                TPview2[(i + dy1), (j + dx1), 0] = 255
                                TPview2[(i + dy1), (j + dx1), 1] = 0
                                TPview2[(i + dy1), (j + dx1), 2] = 0
                    # print(x1,y1,x4,y4)
                    # print("Damage Location 2nd: ",dx1,dy1,dx4,dy4)

        parts = [front_parts, right_parts, left_parts, rear_parts]
        side = "None"
        cnt = 0
        TPview2 = cv2.cvtColor(
            mpimg.imread(
                os.path.join(
                    self.maskBlendImgDir,
                    "no_damage.jpg",
                )
            ),
            cv2.COLOR_BGR2RGB,
        )
        for i in parts:
            cnt += 1
            if len(i) != 0:
                if len(i) == max(
                    [
                        len(front_parts),
                        len(rear_parts),
                        len(left_parts),
                        len(right_parts),
                    ]
                ):
                    # print("count", cnt)
                    if cnt == 1:
                        side = "Front"
                    if cnt == 2:
                        side = "Right"
                    if cnt == 3:
                        side = "Left"
                    if cnt == 4:
                        side = "Rear"

        # print("side",side)

        # To find major damage depend on detected damages.
        major_damage = "None"
        if major_damage == "None":
            for k, v in final_dict.items():
                if len(v) != 0 and k != "Unidentified Panel":
                    for i in v:
                        for a, b in i.items():
                            if a == "bumper replace":
                                major_damage = a
                            elif a == "replace" and major_damage != "bumper replace":
                                major_damage = a
                            elif (
                                a == "dent major"
                                and major_damage != "replace"
                                and major_damage != "bumper replace"
                            ):
                                major_damage = a
                            elif (
                                a == "dent medium"
                                and major_damage != "dent major"
                                and major_damage != "replace"
                                and major_damage != "bumper replace"
                            ):
                                major_damage = a
                            elif (
                                a == "bumper dent"
                                and major_damage != "dent medium"
                                and major_damage != "dent major"
                                and major_damage != "replace"
                                and major_damage != "bumper replace"
                            ):
                                major_damage = a
                            elif (
                                a == "bumper fix"
                                and major_damage != "bumper dent"
                                and major_damage != "dent medium"
                                and major_damage != "dent major"
                                and major_damage != "replace"
                                and major_damage != "bumper replace"
                            ):
                                major_damage = a
                            elif (
                                a == "dent minor"
                                and major_damage != "bumper fix"
                                and major_damage != "bumper dent"
                                and major_damage != "dent medium"
                                and major_damage != "dent major"
                                and major_damage != "replace"
                                and major_damage != "bumper replace"
                            ):
                                major_damage = a
                            elif (
                                a == "crack"
                                and major_damage != "dent minor"
                                and major_damage != "bumper fix"
                                and major_damage != "bumper dent"
                                and major_damage != "dent medium"
                                and major_damage != "dent major"
                                and major_damage != "replace"
                                and major_damage != "bumper replace"
                            ):
                                major_damage = a
                            elif (
                                a == "scratch"
                                and major_damage != "crack"
                                and major_damage != "dent minor"
                                and major_damage != "bumper fix"
                                and major_damage != "bumper dent"
                                and major_damage != "dent medium"
                                and major_damage != "dent major"
                                and major_damage != "replace"
                                and major_damage != "bumper replace"
                            ):
                                major_damage = a
                            else:
                                continue
        # print('major_damage',major_damage)
        # print('---------')
        #'''
        panel_list = []
        for k, v in final_dict.items():
            if len(v) != 0 and k != "Unidentified Panel":
                for i in v:
                    for a, b in i.items():
                        if a == major_damage:
                            if (
                                k != "front grill"
                                and k != "left running board"
                                and k != "left indicator"
                                and k != "right indicator"
                                and k != "left orvm"
                                and k != "right orvm"
                                and k != "left running board"
                                and k != "right running board"
                                and k != "left running board"
                            ):
                                panel_list.append(k)
                                # print(k,'=',majar_damage)
        panel_list = set(panel_list)
        # print(panel_list,'=', major_damage)
        # print("side",side)
        if side == "Left":
            for i in panel_list:
                temp = i.split(" ")
                if temp[0] == "left":
                    panel_list.clear()
                    # panel_list.append(i)
                    panel_list.add(i)
                    break
        if side == "Right":
            for i in panel_list:
                temp = i.split(" ")
                if temp[0] == "right":
                    panel_list.clear()
                    # panel_list.append(i)
                    panel_list.add(i)
                    break
        # print(panel_list,'=', major_damage)
        # print("majar_damage",majar_damage)
        panel_names = [
            "left fender",
            "left front door",
            "left quarter panel",
            "left rear door",
            "right fender",
            "right front door",
            "right quarter panel",
            "right rear door",
            "bonnet",
            "front bumper",
            "right headlamp",
            "left headlamp",
            "boot",
            "rear bumper",
            "right tail lamp",
            "left tail lamp",
        ]

        TPview2_images = [
            "left fender.jpg",
            "left front door.jpg",
            "left quarter panel.jpg",
            "left rear door.jpg",
            "right fender.jpg",
            "right front door.jpg",
            "right quarter panel.jpg",
            "right rear door.jpg",
            "front bumper.jpg",
            "front bumper.jpg",
            "right headlamp.jpg",
            "left headlamp.jpg",
            "rear bumper.jpg",
            "rear bumper.jpg",
            "right tail lamp.jpg",
            "left tail lamp.jpg",
        ]
        panel_damage_locations = [
            [326, 65, 336, 130],
            [329, 147, 337, 206],
            [315, 288, 334, 333],
            [328, 208, 337, 290],
            [323, 271, 335, 322],
            [328, 193, 336, 248],
            [318, 68, 335, 109],
            [326, 110, 337, 192],
            [356, 153, 420, 234],
            [419, 146, 430, 249],
            [395, 128, 412, 148],
            [399, 242, 414, 262],
            [391, 148, 414, 255],
            [416, 148, 426, 255],
            [402, 263, 419, 275],
            [399, 131, 419, 140],
        ]
        maxx = "no_damage.jpg"
        for i in panel_list:
            for j in panel_names:
                if i == j:
                    position = panel_names.index(j)
                    # print(position)
                    TPview2 = cv2.cvtColor(
                        mpimg.imread(
                            os.path.join(
                                self.maskBlendImgDir,
                                TPview2_images[position],
                            )
                        ),
                        cv2.COLOR_BGR2RGB,
                    )
                    maxx = str(TPview2_images[position])
                    for k, v in final_dict.items():
                        if len(v) != 0 and k != "Unidentified Panel":
                            for i in v:
                                for a, b in i.items():
                                    index = panel_label.index(k)
                                    index2 = damage_score.index(b)

                                    TPview2 = cv2.rotate(
                                        TPview2, cv2.ROTATE_90_CLOCKWISE
                                    )
                                    if panel_label[index] == j:
                                        mapping2(
                                            index,
                                            index2,
                                            panel_damage_locations[position][0],
                                            panel_damage_locations[position][1],
                                            panel_damage_locations[position][2],
                                            panel_damage_locations[position][3],
                                        )
                                    TPview2 = cv2.rotate(
                                        TPview2, cv2.ROTATE_90_COUNTERCLOCKWISE
                                    )

                    break
                else:
                    continue
            break
        # Adding TPview2 on top of the TPview

        TPview = TPview[50:]
        TPview = np.concatenate((TPview2, TPview), axis=0)
        # print(TPview.shape)
        TPview = TPview[400:]
        TPview = self.image_resize(TPview, height=800)
        # print(TPview.shape)

        file = open(self.result_dir + "all_final_dict.txt", "a")
        file.writelines("\n" + str(final_dict))
        file.close()

        return (
            outimg,
            dmgMaskImg,
            outClass,
            panel_label,
            TPview,
            final_dict,
            damageTypeOfCar,
        )  # Panel and damage map image, Damage mask only image, panels detected, Final topview(Blueprint),
        # Dictionary of damge locations and damages, Type of damage(minor, medium, major)

    def both_car_max_damage_part(self):
        file = open(self.result_dir + "all_final_dict.txt", "r+")
        one = []
        two = []

        for i in file:
            if i != "\n":
                i = ast.literal_eval(i)
                if len(one) <= 3:
                    one.append(i)
                elif len(two) <= 3:
                    two.append(i)
                else:
                    continue

        both_car = [one, two]
        car1_value = []
        car2_value = []
        for x in both_car:
            for i in x:
                to_remove = []
                for k, v in i.items():
                    if len(v) != 0 and k != "Unidentified Panel":
                        if x == one:
                            car1_value.append(v)
                        else:
                            car2_value.append(v)
                    else:
                        to_remove.append(k)
                # print(to_remove)
                for j in to_remove:
                    del i[j]
        # print(car1_value)
        # print(car2_value)

        major_damage_part_car1 = []
        major_damage_part_car2 = []
        both_car = [car1_value, car2_value]
        # To find major damage depend on detected damages.
        major_dmg_lst = []
        for d in both_car:
            major_damage = "None"
            if major_damage == "None":
                for k in d:
                    for l in k:
                        for a, b in l.items():
                            # print(a)
                            if a == "bumper replace":
                                major_damage = a
                            elif a == "replace" and major_damage != "bumper replace":
                                major_damage = a
                            elif (
                                a == "dent major"
                                and major_damage != "replace"
                                and major_damage != "bumper replace"
                            ):
                                major_damage = a
                            elif (
                                a == "dent medium"
                                and major_damage != "dent major"
                                and major_damage != "replace"
                                and major_damage != "bumper replace"
                            ):
                                major_damage = a
                            elif (
                                a == "bumper dent"
                                and major_damage != "dent medium"
                                and major_damage != "dent major"
                                and major_damage != "replace"
                                and major_damage != "bumper replace"
                            ):
                                major_damage = a
                            elif (
                                a == "bumper fix"
                                and major_damage != "bumper dent"
                                and major_damage != "dent medium"
                                and major_damage != "dent major"
                                and major_damage != "replace"
                                and major_damage != "bumper replace"
                            ):
                                major_damage = a
                            elif (
                                a == "dent minor"
                                and major_damage != "bumper fix"
                                and major_damage != "bumper dent"
                                and major_damage != "dent medium"
                                and major_damage != "dent major"
                                and major_damage != "replace"
                                and major_damage != "bumper replace"
                            ):
                                major_damage = a
                            elif (
                                a == "crack"
                                and major_damage != "dent minor"
                                and major_damage != "bumper fix"
                                and major_damage != "bumper dent"
                                and major_damage != "dent medium"
                                and major_damage != "dent major"
                                and major_damage != "replace"
                                and major_damage != "bumper replace"
                            ):
                                major_damage = a
                            elif (
                                a == "scratch"
                                and major_damage != "crack"
                                and major_damage != "dent minor"
                                and major_damage != "bumper fix"
                                and major_damage != "bumper dent"
                                and major_damage != "dent medium"
                                and major_damage != "dent major"
                                and major_damage != "replace"
                                and major_damage != "bumper replace"
                            ):
                                major_damage = a
                            else:
                                continue
                # print("-",major_damage)
                major_dmg_lst.append(major_damage)
        for i in one:
            for k, v in i.items():
                for l in v:
                    for a, b in l.items():
                        # print(a)
                        if a == major_dmg_lst[0]:
                            major_damage_part_car1.append(k)
        for i in two:
            for k, v in i.items():
                for l in v:
                    for a, b in l.items():
                        # print(a)
                        if a == major_dmg_lst[1]:
                            major_damage_part_car2.append(k)

        file.close()
        # print((major_damage_part_car1))
        # print((major_damage_part_car2))
        file = open(self.result_dir + "all_final_dict.txt", "r+")
        # file.truncate()
        file.close()

        return major_damage_part_car1[0], major_damage_part_car2[0]

    def composite_two_car(
        self, image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8
    ):
        # car1_part, car2_part is max damage part name from 4 and another 4 image
        car1_part, car2_part = self.both_car_max_damage_part()

        img1 = image_1.copy()
        img2 = image_2.copy()
        img3 = image_3.copy()
        img4 = image_4.copy()

        img5 = image_5.copy()
        img6 = image_6.copy()
        img7 = image_7.copy()
        img8 = image_8.copy()

        def red_color_increase(data):
            data = data[280:]
            height, width, depth = data.shape
            for i in range(height):
                for j in range(width):
                    if (
                        data[i, j, 0] >= 240
                        and data[i, j, 1] <= 210
                        and data[i, j, 2] <= 210
                    ):
                        data[i, j, 0] = 255
                        data[i, j, 1] = 50
                        data[i, j, 2] = 50
            return data

        def merge_four_image(img1, img2, img3, img4):
            alpha = 0.5
            beta = 1.0 - alpha
            img1 = cv2.addWeighted(img1, alpha, img2, beta, 0.0)
            red_color_increase(img1)
            img1 = cv2.addWeighted(img1, alpha, img3, beta, 0.0)
            red_color_increase(img1)
            img1 = cv2.addWeighted(img1, alpha, img4, beta, 0.0)
            red_color_increase(img1)
            return img1

        final_top_view1 = merge_four_image(img1, img2, img3, img4)
        final_top_view2 = merge_four_image(img5, img6, img7, img8)
        # _ = plt.imshow(final_top_view1)
        # _ = plt.imshow(final_top_view2)

        for f_Img in (final_top_view1, final_top_view2):
            height, width, depth = f_Img.shape
            for i in range(280):
                for j in range(width):
                    f_Img[i, j, 0] = 255
                    f_Img[i, j, 1] = 255
                    f_Img[i, j, 2] = 255

        c1 = final_top_view1.copy()
        c1 = c1[292:448, 95:440]
        c1 = self.image_resize(c1, height=110)
        # c1 = Image.fromarray(c1)
        # _ = plt.imshow(c1)

        c2 = final_top_view2.copy()
        c2 = c2[292:448, 95:440]
        c2 = self.image_resize(c2, height=110)
        # c2 = Image.fromarray(c2)
        # _ = plt.imshow(c2)
        # print(c1.shape)
        # print(c2.shape)

        number_one = cv2.cvtColor(
            mpimg.imread(os.path.join(self.maskBlendImgDir, "number one.jpg")),
            cv2.COLOR_BGR2RGB,
        )
        number_two = cv2.cvtColor(
            mpimg.imread(os.path.join(self.maskBlendImgDir, "number two.jpg")),
            cv2.COLOR_BGR2RGB,
        )

        number_one = self.image_resize(number_one, height=40)
        # number_one = cv2.rotate(number_one, cv2.ROTATE_90_CLOCKWISE)

        number_two = self.image_resize(number_two, height=40)
        # number_two = cv2.rotate(number_two, cv2.ROTATE_90_CLOCKWISE)

        # car1_part = 'left fender'
        # car2_part = 'rear bumper'
        if (
            car1_part == "bonnet"
            or car1_part == "front bumper"
            or car1_part == "left headlamp"
            or car1_part == "right headlamp"
        ):
            number_one = cv2.rotate(number_one, cv2.ROTATE_90_CLOCKWISE)
            number_one = cv2.rotate(number_one, cv2.ROTATE_90_CLOCKWISE)
        if (
            car2_part == "bonnet"
            or car2_part == "front bumper"
            or car2_part == "left headlamp"
            or car2_part == "right headlamp"
        ):
            number_two = cv2.rotate(number_two, cv2.ROTATE_90_CLOCKWISE)
            number_two = cv2.rotate(number_two, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if (
            car1_part == "boot"
            or car1_part == "rear bumper"
            or car1_part == "left tail lamp"
            or car1_part == "right tail lamp"
        ):
            number_one = cv2.rotate(number_one, cv2.ROTATE_90_CLOCKWISE)
            number_one = cv2.rotate(number_one, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if (
            car2_part == "boot"
            or car2_part == "rear bumper"
            or car2_part == "left tail lamp"
            or car2_part == "right tail lamp"
        ):
            number_two = cv2.rotate(number_two, cv2.ROTATE_90_CLOCKWISE)
            number_two = cv2.rotate(number_two, cv2.ROTATE_90_CLOCKWISE)

        if (
            car1_part == "right fender"
            or car1_part == "right front door"
            or car1_part == "right rear door"
            or car1_part == "right quarter panel"
        ):
            number_one = cv2.rotate(number_one, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if (
            car2_part == "right fender"
            or car2_part == "right front door"
            or car2_part == "right rear door"
            or car2_part == "right quarter panel"
        ):
            number_two = cv2.rotate(number_two, cv2.ROTATE_90_CLOCKWISE)

        if (
            car1_part == "left fender"
            or car1_part == "left front door"
            or car1_part == "left rear door"
            or car1_part == "left quarter panel"
        ):
            number_one = cv2.rotate(number_one, cv2.ROTATE_90_CLOCKWISE)
        if (
            car2_part == "left fender"
            or car2_part == "left front door"
            or car2_part == "left rear door"
            or car2_part == "left quarter panel"
        ):
            number_two = cv2.rotate(number_two, cv2.ROTATE_90_COUNTERCLOCKWISE)

        number_one = Image.fromarray(number_one)
        number_two = Image.fromarray(number_two)

        c1 = Image.fromarray(c1)
        c2 = Image.fromarray(c2)

        c1.paste(number_one, (120, 37))
        c2.paste(number_two, (120, 37))

        c1 = np.array(c1)
        c2 = np.array(c2)

        final_top_view1 = Image.fromarray(final_top_view1)
        final_top_view2 = Image.fromarray(final_top_view2)

        def paste_collide_image(c1, c2, x1, y1, x2, y2):
            c1 = Image.fromarray(c1)
            c2 = Image.fromarray(c2)

            final_top_view1.paste(c1, (x1, y1))
            final_top_view1.paste(c2, (x2, y2))

            final_top_view2.paste(c1, (x1, y1))
            final_top_view2.paste(c2, (x2, y2))

            return final_top_view1, final_top_view2

        # car1_part = 'boot'
        # car2_part = 'right front '

        # Car 1 possible Positions
        # front
        if (
            car1_part == "front bumper"
            or car1_part == "bonnet"
            or car1_part == "right headlamp"
            or car1_part == "left headlamp"
        ) and (car2_part == "front bumper" or car2_part == "bonnet"):
            c1 = cv2.rotate(c1, cv2.ROTATE_180)
            paste_collide_image(c1, c2, 20, 80, 260, 80)
        # right
        elif car1_part == "right fender" and (
            car2_part == "front bumper" or car2_part == "bonnet"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_90_CLOCKWISE)
            paste_collide_image(c1, c2, 150, 30, 253, 30)
        elif car1_part == "right front door" and (
            car2_part == "front bumper" or car2_part == "bonnet"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_90_CLOCKWISE)
            paste_collide_image(c1, c2, 150, 30, 253, 80)
        elif car1_part == "right rear door" and (
            car2_part == "front bumper" or car2_part == "bonnet"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_90_CLOCKWISE)
            paste_collide_image(c1, c2, 150, 30, 253, 130)
        elif car1_part == "right quarter panel" and (
            car2_part == "front bumper" or car2_part == "bonnet"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_90_CLOCKWISE)
            paste_collide_image(c1, c2, 150, 30, 253, 180)
        # left
        elif car1_part == "left quarter panel" and (
            car2_part == "front bumper" or car2_part == "bonnet"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_90_CLOCKWISE)
            c1 = cv2.rotate(c1, cv2.ROTATE_180)
            paste_collide_image(c1, c2, 150, 30, 253, 30)
        elif car1_part == "left rear door" and (
            car2_part == "front bumper" or car2_part == "bonnet"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_90_CLOCKWISE)
            c1 = cv2.rotate(c1, cv2.ROTATE_180)
            paste_collide_image(c1, c2, 150, 30, 253, 80)
        elif car1_part == "left front door" and (
            car2_part == "front bumper" or car2_part == "bonnet"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_90_CLOCKWISE)
            c1 = cv2.rotate(c1, cv2.ROTATE_180)
            paste_collide_image(c1, c2, 150, 30, 253, 130)
        elif car1_part == "left fender" and (
            car2_part == "front bumper" or car2_part == "bonnet"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_90_CLOCKWISE)
            c1 = cv2.rotate(c1, cv2.ROTATE_180)
            paste_collide_image(c1, c2, 150, 30, 253, 180)

        # rear
        elif (
            car1_part == "boot"
            or car1_part == "rear bumper"
            or car1_part == "right tail lamp"
            or car1_part == "left tail lamp"
        ) and (car2_part == "front bumper" or car2_part == "bonnet"):
            paste_collide_image(c1, c2, 20, 80, 260, 80)

        # front
        elif (
            car1_part == "front bumper"
            or car1_part == "bonnet"
            or car1_part == "right headlamp"
            or car1_part == "left headlamp"
        ) and (car2_part == "boot" or car2_part == "rear bumper"):
            c1 = cv2.rotate(c1, cv2.ROTATE_180)
            c2 = cv2.rotate(c2, cv2.ROTATE_180)
            paste_collide_image(c1, c2, 20, 80, 260, 80)

        # right
        elif car1_part == "right fender" and (
            car2_part == "boot" or car2_part == "rear bumper"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_90_CLOCKWISE)
            c2 = cv2.rotate(c2, cv2.ROTATE_180)
            paste_collide_image(c1, c2, 150, 30, 253, 30)
        elif car1_part == "right front door" and (
            car2_part == "boot" or car2_part == "rear bumper"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_90_CLOCKWISE)
            c2 = cv2.rotate(c2, cv2.ROTATE_180)
            paste_collide_image(c1, c2, 150, 30, 253, 80)
        elif car1_part == "right rear door" and (
            car2_part == "boot" or car2_part == "rear bumper"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_90_CLOCKWISE)
            c2 = cv2.rotate(c2, cv2.ROTATE_180)
            paste_collide_image(c1, c2, 150, 30, 253, 130)
        elif car1_part == "right quarter panel" and (
            car2_part == "boot" or car2_part == "rear bumper"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_90_CLOCKWISE)
            c2 = cv2.rotate(c2, cv2.ROTATE_180)
            paste_collide_image(c1, c2, 150, 30, 253, 180)
        # left
        elif car1_part == "left quarter panel" and (
            car2_part == "boot" or car2_part == "rear bumper"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_90_CLOCKWISE)
            c1 = cv2.rotate(c1, cv2.ROTATE_180)
            c2 = cv2.rotate(c2, cv2.ROTATE_180)
            paste_collide_image(c1, c2, 150, 30, 253, 30)
        elif car1_part == "left rear door" and (
            car2_part == "boot" or car2_part == "rear bumper"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_90_CLOCKWISE)
            c1 = cv2.rotate(c1, cv2.ROTATE_180)
            c2 = cv2.rotate(c2, cv2.ROTATE_180)
            paste_collide_image(c1, c2, 150, 30, 253, 80)
        elif car1_part == "left front door" and (
            car2_part == "boot" or car2_part == "rear bumper"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_90_CLOCKWISE)
            c1 = cv2.rotate(c1, cv2.ROTATE_180)
            c2 = cv2.rotate(c2, cv2.ROTATE_180)
            paste_collide_image(c1, c2, 150, 30, 253, 130)
        elif car1_part == "left fender" and (
            car2_part == "boot" or car2_part == "rear bumper"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_90_CLOCKWISE)
            c1 = cv2.rotate(c1, cv2.ROTATE_180)
            c2 = cv2.rotate(c2, cv2.ROTATE_180)
            paste_collide_image(c1, c2, 150, 30, 253, 180)

        # rear
        elif (
            car1_part == "boot"
            or car1_part == "rear bumper"
            or car1_part == "right tail lamp"
            or car1_part == "left tail lamp"
        ) and (car2_part == "boot" or car2_part == "rear bumper"):
            c2 = cv2.rotate(c2, cv2.ROTATE_180)
            paste_collide_image(c1, c2, 20, 80, 260, 80)

        elif (
            car1_part == "left fender"
            or car1_part == "left front door"
            or car1_part == "left rear door"
            or car1_part == "left quarter panel"
        ) and (
            car2_part == "left fender"
            or car2_part == "left front door"
            or car2_part == "left rear door"
            or car2_part == "left quarter panel"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_90_CLOCKWISE)
            c1 = cv2.rotate(c1, cv2.ROTATE_180)
            # c2 = cv2.rotate(c2, cv2.ROTATE_90_COUNTERCLOCKWISE)
            c2 = cv2.rotate(c2, cv2.ROTATE_90_CLOCKWISE)
            paste_collide_image(c1, c2, 150, 30, 253, 30)

        elif (
            car1_part == "right fender"
            or car1_part == "right front door"
            or car1_part == "right rear door"
            or car1_part == "right quarter panel"
        ) and (
            car2_part == "right fender"
            or car2_part == "right front door"
            or car2_part == "right rear door"
            or car2_part == "right quarter panel"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_90_COUNTERCLOCKWISE)
            c1 = cv2.rotate(c1, cv2.ROTATE_180)
            c2 = cv2.rotate(c2, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # c2 = cv2.rotate(c2, cv2.ROTATE_90_CLOCKWISE)
            paste_collide_image(c1, c2, 150, 30, 253, 30)
        elif (
            car1_part == "left fender"
            or car1_part == "left front door"
            or car1_part == "left rear door"
            or car1_part == "left quarter panel"
        ) and (
            car2_part == "right fender"
            or car2_part == "right front door"
            or car2_part == "left rear door"
            or car2_part == "right quarter panel"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_90_CLOCKWISE)
            c1 = cv2.rotate(c1, cv2.ROTATE_180)
            c2 = cv2.rotate(c2, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # c2 = cv2.rotate(c2, cv2.ROTATE_90_CLOCKWISE)
            paste_collide_image(c1, c2, 150, 30, 253, 30)

        elif (
            car1_part == "right fender"
            or car1_part == "right front door"
            or car1_part == "right rear door"
            or car1_part == "right quarter panel"
        ) and (
            car2_part == "left fender"
            or car2_part == "left front door"
            or car2_part == "left rear door"
            or car2_part == "left quarter panel"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_90_COUNTERCLOCKWISE)
            c1 = cv2.rotate(c1, cv2.ROTATE_180)
            # c2 = cv2.rotate(c2, cv2.ROTATE_90_COUNTERCLOCKWISE)
            c2 = cv2.rotate(c2, cv2.ROTATE_90_CLOCKWISE)
            paste_collide_image(c1, c2, 150, 30, 253, 30)

        # Car 2 possible Positions
        # right
        elif car2_part == "right fender" and (
            car1_part == "front bumper" or car1_part == "bonnet"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_180)
            c2 = cv2.rotate(c2, cv2.ROTATE_90_COUNTERCLOCKWISE)
            paste_collide_image(c1, c2, 60, 170, 300, 40)
        elif car2_part == "right front door" and (
            car1_part == "front bumper" or car1_part == "bonnet"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_180)
            c2 = cv2.rotate(c2, cv2.ROTATE_90_COUNTERCLOCKWISE)
            paste_collide_image(c1, c2, 60, 130, 300, 40)
        elif car2_part == "right rear door" and (
            car1_part == "front bumper" or car1_part == "bonnet"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_180)
            c2 = cv2.rotate(c2, cv2.ROTATE_90_COUNTERCLOCKWISE)
            paste_collide_image(c1, c2, 60, 90, 300, 40)
        elif car2_part == "right quarter panel" and (
            car1_part == "front bumper" or car1_part == "bonnet"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_180)
            c2 = cv2.rotate(c2, cv2.ROTATE_90_COUNTERCLOCKWISE)
            paste_collide_image(c1, c2, 60, 40, 300, 40)
        # left
        elif car2_part == "left quarter panel" and (
            car1_part == "front bumper" or car1_part == "bonnet"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_180)
            c2 = cv2.rotate(c2, cv2.ROTATE_90_CLOCKWISE)
            paste_collide_image(c1, c2, 60, 180, 300, 40)
        elif car2_part == "left rear door" and (
            car1_part == "front bumper" or car1_part == "bonnet"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_180)
            c2 = cv2.rotate(c2, cv2.ROTATE_90_CLOCKWISE)
            paste_collide_image(c1, c2, 60, 140, 300, 40)
        elif car2_part == "left front door" and (
            car1_part == "front bumper" or car1_part == "bonnet"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_180)
            c2 = cv2.rotate(c2, cv2.ROTATE_90_CLOCKWISE)
            paste_collide_image(c1, c2, 60, 100, 300, 40)
        elif car2_part == "left fender" and (
            car1_part == "front bumper" or car1_part == "bonnet"
        ):
            c1 = cv2.rotate(c1, cv2.ROTATE_180)
            c2 = cv2.rotate(c2, cv2.ROTATE_90_CLOCKWISE)
            paste_collide_image(c1, c2, 60, 50, 300, 40)

        # right
        elif car2_part == "right fender" and (
            car1_part == "rear bumper" or car1_part == "boot"
        ):
            # c1 = cv2.rotate(c1, cv2.ROTATE_180)
            c2 = cv2.rotate(c2, cv2.ROTATE_90_COUNTERCLOCKWISE)
            paste_collide_image(c1, c2, 60, 170, 300, 40)
        elif car2_part == "right front door" and (
            car1_part == "rear bumper" or car1_part == "boot"
        ):
            # c1 = cv2.rotate(c1, cv2.ROTATE_180)
            c2 = cv2.rotate(c2, cv2.ROTATE_90_COUNTERCLOCKWISE)
            paste_collide_image(c1, c2, 60, 130, 300, 40)
        elif car2_part == "right rear door" and (
            car1_part == "rear bumper" or car1_part == "boot"
        ):
            # c1 = cv2.rotate(c1, cv2.ROTATE_180)
            c2 = cv2.rotate(c2, cv2.ROTATE_90_COUNTERCLOCKWISE)
            paste_collide_image(c1, c2, 60, 90, 300, 40)
        elif car2_part == "right quarter panel" and (
            car1_part == "rear bumper" or car1_part == "boot"
        ):
            # c1 = cv2.rotate(c1, cv2.ROTATE_180)
            c2 = cv2.rotate(c2, cv2.ROTATE_90_COUNTERCLOCKWISE)
            paste_collide_image(c1, c2, 60, 40, 300, 40)
        # left
        elif car2_part == "left quarter panel" and (
            car1_part == "rear bumper" or car1_part == "boot"
        ):
            # c1 = cv2.rotate(c1, cv2.ROTATE_180)
            c2 = cv2.rotate(c2, cv2.ROTATE_90_CLOCKWISE)
            paste_collide_image(c1, c2, 60, 180, 300, 40)
        elif car2_part == "left rear door" and (
            car1_part == "rear bumper" or car1_part == "boot"
        ):
            # c1 = cv2.rotate(c1, cv2.ROTATE_180)
            c2 = cv2.rotate(c2, cv2.ROTATE_90_CLOCKWISE)
            paste_collide_image(c1, c2, 60, 140, 300, 40)
        elif car2_part == "left front door" and (
            car1_part == "rear bumper" or car1_part == "boot"
        ):
            # c1 = cv2.rotate(c1, cv2.ROTATE_180)
            c2 = cv2.rotate(c2, cv2.ROTATE_90_CLOCKWISE)
            paste_collide_image(c1, c2, 60, 100, 300, 40)
        elif car2_part == "left fender" and (
            car1_part == "rear bumper" or car1_part == "boot"
        ):
            # c1 = cv2.rotate(c1, cv2.ROTATE_180)
            c2 = cv2.rotate(c2, cv2.ROTATE_90_CLOCKWISE)
            paste_collide_image(c1, c2, 60, 50, 300, 40)
        else:
            c1 = cv2.rotate(c1, cv2.ROTATE_90_CLOCKWISE)
            # c1 = cv2.rotate(c1, cv2.ROTATE_180)
            # c2 = cv2.rotate(c2, cv2.ROTATE_90_COUNTERCLOCKWISE)
            c2 = cv2.rotate(c2, cv2.ROTATE_90_CLOCKWISE)
            paste_collide_image(c1, c2, 130, 30, 270, 30)

        number_one = cv2.cvtColor(
            mpimg.imread(os.path.join(self.maskBlendImgDir, "number one.jpg")),
            cv2.COLOR_BGR2RGB,
        )
        number_two = cv2.cvtColor(
            mpimg.imread(os.path.join(self.maskBlendImgDir, "number two.jpg")),
            cv2.COLOR_BGR2RGB,
        )

        number_one = self.image_resize(number_one, height=40)
        number_two = self.image_resize(number_two, height=40)

        number_one = Image.fromarray(number_one)
        number_two = Image.fromarray(number_two)

        final_top_view1.paste(number_one, (270, 350))
        final_top_view2.paste(number_two, (270, 350))
        # final_top_view1 = Image.fromarray(final_top_view1)
        # final_top_view2 = Image.fromarray(final_top_view2)

        both_car = final_top_view1.crop((0, 0, 572, 290))  # this image contain both car
        final_top_view1 = final_top_view1.crop((0, 285, 572, 800))
        final_top_view2 = final_top_view2.crop((0, 285, 572, 800))

        return final_top_view1, final_top_view2, both_car

    def process_image(self, image_file):
        im = cv2.imread(image_file)
        # Aspect ratio preserved image resize
        im = self.image_resize(im, height=800)
        (
            img,
            dmgMaskImg,
            outClass,
            panel_label,
            TPview,
            final_dict,
            damageTypeOfCar,
        ) = self.get_prediction(im)
        return (
            img,
            dmgMaskImg,
            outClass,
            panel_label,
            TPview,
            final_dict,
            damageTypeOfCar,
        )
