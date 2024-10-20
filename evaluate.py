#!/usr/bin/env python

import pathlib
import argparse
import numpy as np
import yaml
import torch
import tqdm
import wandb
from gaze_estimation import (GazeEstimationMethod, create_testloader,
                             create_model)
from gaze_estimation.utils import compute_angle_error, load_config, save_config
from test_loader import get_test_loader
#from evaluation import evaluate_metrics  # if have a evaluation


class GazeTest:
    def __init__(self, model, config):
        """
        init GazeTest
        :param model: 需要测试的模型
        :param config: 配置信息
        """
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if config.test.use_gpu and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.load_mode = self.get_load_mode(config)  # 根据模型和数据集获取加载模式
        self.test_loader = self.load_data()
        wandb.init(project='project-name', mode="disabled")  # 禁用 WandB 日志记录，可根据需要启用
    
    def get_load_mode(self, config):
        """
        根据模型名称或数据集类型选择加载模式
        :param config: 配置信息
        :return: 加载模式
        """
        if config.dataset.name == "EVE":
            return "load_eve"
        elif config.dataset.name == "GAZE360":
            return "load_gaze360"
        elif config.dataset.name == "XGAZE":
            return "load_xgaze"
        elif config.dataset.name == "MPII":
            return "load_mpii"
        # more dataset
        else:
            raise ValueError(f"Unknown dataset: {config.dataset.name}")
    
    def load_data(self):
        """
        根据配置信息加载测试数据集
        :return: 数据加载器
        """
        #TODO: def create_dataloader(data_dir, batch_size, load_mode, is_shuffle=False, num_workers)
        if self.load_mode == "load_mpii" or self.load_mode == "load_gaze360":
            test_loader = create_testloader(self.config, is_train=False)

        elif self.load_mode == "load_xgaze":
            #data_dir, batch_size, load_mode, num_workers=4, is_shuffle=True
            test_loader = get_test_loader(self.config.data_dir, 
                self.config.batch_size, 
                self.load_mode,  
                num_workers=self.config.num_workers,
                is_shuffle=False
                )
        # else:
        #     test_loader = create_testloader(
        #         self.config.data_dir, 
        #         self.config.batch_size, 
        #         self.load_mode,  
        #         num_workers=self.config.num_workers,
        #         is_shuffle=False
        #     )
        return test_loader
        
    def test(self):
        """
        测试模型在指定数据集上的表现
        """
        #加载模型权重（建议在GazeTest传入model前完成）
        # model = create_model(self.config)
        # checkpoint = torch.load(self.config.test.checkpoint, map_location='cpu')
        # model.load_state_dict(checkpoint['model'])
        
        #输出路径设置
        output_rootdir = pathlib.Path(self.config.test.output_dir)
        checkpoint_name = pathlib.Path(self.config.model.checkpoint).stem
        output_dir = output_rootdir / checkpoint_name
        output_dir.mkdir(exist_ok=True, parents=True)
        save_config(self.config, output_dir)


        # 加载数据
        test_loader = self.test_loader

        # 模型进入评估模式
        self.model.eval()
        
        # 用于保存测试结果的变量
        predictions = []
        gts = [] 
        if self.load_mode == 'load_mpii':
            with torch.no_grad():
                for images, gazes in tqdm.tqdm(test_loader):

                    #REMOVE BELOW
                    # if load_mode == 'load_single_face':
                    #     image = images['face'].to(self.device)
                    #     gazes = gazes.to(self.device)
                    #     outputs = self.model(image)

                    image = images['face'].to(self.device)
                    gazes = gazes.to(self.device)
                    outputs = self.model(image)
                    predictions.append(outputs.cpu())
                    gts.append(gazes.cpu())
            predictions = torch.cat(predictions)
            gts = torch.cat(gts)
            metric = float(compute_angle_error(predictions, gts).mean())
        else:
            predictions = []
            gts = []
            with torch.no_grad():  # 不需要梯度计算
                for batch_idx, (images, gazes) in tqdm.tqdm(test_loader):
                    images, gazes = images.to(self.device), gazes.to(self.device)
                    
                    # 模型预测
                    outputs = self.model(images)
                    predictions.append(outputs.cpu())
                    gts.append(gazes.cpu())
                    # 将结果记录下来
                    # if batch_idx % self.config.print_freq == 0:
                    #     print(f"Batch {batch_idx}/{len(test_loader)} tested.")
            
            # 评估指标计算
            metric, results = self.evaluate((predictions, gts))
        return predictions, gts, metric
    
    def evaluate(self, results):
        """
        评估模型的输出
        :param results: 模型的预测结果和真实标签
        :return: 评估结果
        """
        # 根据不同数据集或任务设计不同的评估指标，比如 AUC, NSS 等
        #if config.test.metric = 'xxxx':
        #   return xxx(results)    


        metric = float(compute_angle_error(results[0],results[1]).mean())  # 传入你定义的评估函数
        return metric, results




"""
below from MPIIFaceGaze
"""
def test(model, test_loader, config):
    model.eval()
    device = torch.device(config.device)
    if config.model.name == 'face_res50':
        load_mode = 'load_single_face'
    else:
        raise Exception("Please enter a correct model name or choose a correct load mode for your model (load_single_face or load_multi_region).")

    predictions = []
    gts = []
    with torch.no_grad():
        for images, gazes in tqdm.tqdm(test_loader):
            if load_mode == 'load_single_face':
                image = images['face'].to(device)
                gazes = gazes.to(device)
                outputs = model(image)
            predictions.append(outputs.cpu())
            gts.append(gazes.cpu())
    predictions = torch.cat([predictions,outputs])
    gts = torch.cat(gts)
    angle_error = float(compute_angle_error(predictions, gts).mean())
    return predictions, gts, angle_error




#REMOVED
# def load_config():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str)
#     parser.add_argument('--datase_name', type=str, default=None)
#     parser.add_argument('--model_stride', type=int, default=None)
#     parser.add_argument('--data_dir', type=str, default=None)
#     parser.add_argument('--size', type=int, default=None)
#     parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
#     args = parser.parse_args()
#     if args.config is not None:
#         with open(args.config, 'r') as file:
#             config = yaml.safe_load(file)
#     return config



def main():
    config = load_config()

    print(config)


    # tput_rootdir = pathlib.Path(config.test.output_dir)
    # checkpoint_name = pathlib.Path(config.test.checkpoint).stem
    # output_dir = output_rootdir / checkpoint_name
    # output_dir.mkdir(exist_ok=True, parents=True)
    # save_config(config, output_dir)

    test_loader = create_testloader(config, is_train=False)

    model = create_model(config)
    checkpoint = torch.load(config.model.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    MPIITest = GazeTest(model, config)
    predictions, gts, angle_error = MPIITest.test()
    print(f'The mean angle error (deg): {angle_error:.2f}')
if __name__ == '__main__':
    main()
