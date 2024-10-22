from typing import Any

import cv2
import numpy as np
import torch
import torchvision
import yacs.config

from .types import GazeEstimationMethod


def create_transform(config: yacs.config.CfgNode) -> Any:
    return _create_mpiifacegaze_transform(config)
    

def _create_mpiifacegaze_transform(config: yacs.config.CfgNode) -> Any:
    scale = torchvision.transforms.Lambda(lambda x: x.astype(np.float32) / 255)
    identity = torchvision.transforms.Lambda(lambda x: x)
    size = config.transform.mpiifacegaze_face_size

    if config.model.name == 'face_res50':
        load_mode = 'load_single_face'
    elif config.model.name == 'multi_region_res50':
        load_mode = 'load_multi_region'
    elif config.model.name == 'multi_region_res50_share_eyenet':
        load_mode = 'load_multi_region'
    else:
        raise Exception("Please enter a correct model name or choose a correct load mode for your model (load_single_face or load_multi_region).")

    if load_mode == 'load_single_face':
        if size != 448:
            resize = torchvision.transforms.Lambda(
                lambda x: cv2.resize(x, (size, size)))
        else:
            resize = identity
    else:
        resize = identity
    if config.transform.mpiifacegaze_gray:
        to_gray = torchvision.transforms.Lambda(lambda x: cv2.cvtColor(
            cv2.equalizeHist(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)), cv2.
            COLOR_GRAY2BGR))
    else:
        to_gray = identity

    """
    We unified the following versions 
    """
    # if config.dataset.name == 'MPII' or config.dataset.name == 'GAZE360' or config.dataset.name == 'ColumbiaGaze':
    #     transform = torchvision.transforms.Compose([
    #         resize,
    #         to_gray,
    #         torchvision.transforms.Lambda(lambda x: x.transpose(2, 0, 1)), #(H,W,C)->(C,H,W)
    #         scale,
    #         torch.from_numpy,
    #         # torchvision.transforms.Normalize(mean=[0.406, 0.456, 0.485],  #for BGR
    #         #                                 std=[0.225, 0.224, 0.229]),
    #         torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], #for RGB
    #                                         std=[0.229, 0.224, 0.225]),

    #     ])
    # elif config.dataset.name == 'GAZE360' or config.dataset.name == 'ColumbiaGaze' :
    #     #TODO(finn): change transform to ensure image format is [3, 224, 224] with RGB  (DONE)
    #     transform = torchvision.transforms.Compose([
    #         resize,
    #         to_gray,
    #         torchvision.transforms.Lambda(lambda x: x.transpose(2, 0, 1)),#(H,W,C)->(C,H,W)
    #         scale,
    #         torch.from_numpy,
    #         torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], #for RGB
    #                                         std=[0.229, 0.224, 0.225]),
    #     ])



    if config.dataset.name == 'XGAZE' :
        transform = torchvision.transforms.Compose([ # 测试数据集的转换操作
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = torchvision.transforms.Compose([
            resize,
            to_gray,
            torchvision.transforms.Lambda(lambda x: x.transpose(2, 0, 1)),#(H,W,C)->(C,H,W)
            scale,
            torch.from_numpy,
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], #for RGB
                                            std=[0.229, 0.224, 0.225]),
        ])

        
    return transform
