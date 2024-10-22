#!/usr/bin/env python

import pathlib
import argparse
import numpy as np
import yaml
import torch
import tqdm
import wandb
from GazeTest import GazeTest
from gaze_estimation import (GazeEstimationMethod, create_testloader,
                             create_model)
from gaze_estimation.utils import compute_angle_error, load_config, save_config
from test_loader import get_test_loader
#from evaluation import evaluate_metrics  # if have a evaluation




def main():
    config = load_config()

    print(config)


    # tput_rootdir = pathlib.Path(config.test.output_dir)
    # checkpoint_name = pathlib.Path(config.test.checkpoint).stem
    # output_dir = output_rootdir / checkpoint_name
    # output_dir.mkdir(exist_ok=True, parents=True)
    # save_config(config, output_dir)

    # test_loader = create_testloader(config, is_train=False)

    model = create_model(config)
    checkpoint = torch.load(config.model.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    Test = GazeTest(model, config)
    if config.dataset.name == "MPII" or config.dataset.name == "GAZE360":
        predictions, gts, angle_error = Test.test()
        print(f'The mean angle error (deg): {angle_error:.2f}')
    elif config.dataset.name == "XGAZE" or config.dataset.name == "EVE":
        predictions = Test.test()
        print(f'The predictions are saved')
    
if __name__ == '__main__':
    main()
