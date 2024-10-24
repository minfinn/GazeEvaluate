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
from utils import pitchyaw_to_vector, to_screen_coordinates2
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
        self.num_test = len(self.test_loader.dataset)
        self.batch_size =  self.config.test.batch_size
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
        elif config.dataset.name == "ColumbiaGaze":
            return "load_columbiagaze"
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
        if self.load_mode == "load_mpii" or self.load_mode == "load_gaze360" or self.load_mode == "load_xgaze" or self.load_mode == "load_columbiagaze" or self.load_mode == "load_eve":
            test_loader = create_testloader(self.config, is_train=False)

        else:
            #data_dir, batch_size, load_mode, num_workers=4, is_shuffle=True
            test_loader = get_test_loader(self.config.dataset.data_dir, 
                self.config.test.batch_size, 
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
    
    def save_test_results(self, output_file_path, model_name, dataset_name, batch_size, num_workers, checkpoint_path, metrics, total_samples):
        """
        保存测试结果到文本文件
        :param output_file_path: 输出文件路径
        :param model_name: 模型名称
        :param dataset_name: 数据集名称
        :param batch_size: 批量大小
        :param num_workers: 数据加载的子进程数
        :param checkpoint_path: 模型权重路径
        :param metrics: 评估指标（如平均角度误差）
        :param total_samples: 总测试样本数
        :param successful_predictions: 成功预测的数量
        :param failed_predictions: 失败预测的数量
        """
        with open(output_file_path, 'w') as f:
            f.write("# Test Configs\n")
            f.write("=========================\n")
            f.write(f"Model Name: {model_name}\n")
            f.write(f"Dataset Name: {dataset_name}\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Number of Workers: {num_workers}\n")
            f.write(f"Checkpoint Path: {checkpoint_path}\n")
            f.write(f"Evaluation Metric: Mean Angle Error\n")
            f.write("\n# Results\n")
            f.write("=========================\n")
            f.write(f"Mean Angle Error (deg): {metrics:.2f}\n")
            f.write(f"Total Test Samples: {total_samples}\n")
            f.write("\n# Additional Notes\n")
            f.write("=========================\n")
            f.write("- The model performed well under various conditions.\n")
            f.write("- Future improvements may include additional data augmentation techniques.\n")

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

        if self.load_mode == 'load_mpii' or self.load_mode == 'load_gaze360' or self.load_mode == 'load_columbiagaze':
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
            output_file_path = output_dir / f'{checkpoint_name}_test_results.txt'
            total_samples = self.num_test
        
            self.save_test_results(
                output_file_path,
                self.config.model.name,
                self.config.dataset.name,
                self.config.test.batch_size,
                self.config.test.dataloader.num_workers,
                self.config.model.checkpoint,
                metric,
                total_samples
            )
            return predictions, gts, metric


        #TODO:trans predictions to screen coordinates and then create a right result_to_save
        if self.load_mode == 'load_eve':
            # template_path = './template.pkl.gz' # result template


            with torch.no_grad():
                for images in tqdm.tqdm(test_loader):

                    image = images['face'].to(self.device)
                    outputs = self.model(image)
                    predictions.append(outputs.cpu())
            print("Prediction is OK")
            predictions = torch.cat(predictions)
            
            preds_3D = pitchyaw_to_vector(predictions)
            output_file_path = output_dir/ f'eve_{checkpoint_name}_test_vector_results.txt'
            np.savetxt(output_file_path, preds_3D, delimiter=',')
            
            
            
        
            return predictions

        elif self.load_mode == 'load_xgaze':
            pred_gaze_all = np.zeros((self.num_test, 2))
            save_index = 0

            for input in tqdm.tqdm(test_loader):
                # print(input.shape)


                face_input_var = torch.autograd.Variable(input.float().cuda())
                pred_gaze= self.model(face_input_var) 
                pred_gaze_all[save_index:save_index+self.batch_size, :] = pred_gaze.cpu().data.numpy()

                save_index += face_input_var.size(0) 

            if save_index != self.num_test:
                print('the test samples save_index ', save_index, ' is not equal to the whole test set ', self.num_test)

            print('Tested on : ', pred_gaze_all.shape[0], ' samples')
            output_file_path = output_dir/ f'within_eva_{checkpoint_name}_test_results.txt'
            np.savetxt(output_file_path, pred_gaze_all, delimiter=',')
            return pred_gaze_all


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
            # 保存测试结果
            output_file_path = output_dir / f'{checkpoint_name}_test_results.txt'
            total_samples = self.num_test
        
            self.save_test_results(
                output_file_path,
                self.config.model.name,
                self.config.dataset.name,
                self.config.test.batch_size,
                self.config.test.dataloader.num_workers,
                self.config.model.checkpoint,
                metric,
                total_samples
            )
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

"""
below from ETH-XGaze
"""

def test(self):
    """
    Test the pre-treained model on the whole test set. Note there is no label released to public, you can
    only save the predicted results. You then need to submit the test resutls to our evaluation website to
    get the final gaze estimation error.
    """
    print('We are now doing the final test')
    self.model.eval()
    self.load_checkpoint(is_strict=False, input_file_path=self.pre_trained_model_path)
    pred_gaze_all = np.zeros((self.num_test, 2))
    mean_error = []
    save_index = 0

    print('Testing on ', self.num_test, ' samples')

    for i, (input) in enumerate(self.test_loader):
        # depending on load mode, input differently
        if self.load_mode == "load_single_face":
            face_input_var = torch.autograd.Variable(input["face"].float().cuda())
            pred_gaze= self.model(face_input_var) 
        elif self.load_mode == "load_multi_region":
            face_input_var = torch.autograd.Variable(input["face"].float().cuda())
            left_eye_input_var = torch.autograd.Variable(input["left_eye"].float().cuda()) 
            right_eye_input_var = torch.autograd.Variable(input["right_eye"].float().cuda())
            pred_gaze= self.model(left_eye_input_var, right_eye_input_var, face_input_var) 

        pred_gaze_all[save_index:save_index+self.batch_size, :] = pred_gaze.cpu().data.numpy()

        save_index += face_input_var.size(0) 

    if save_index != self.num_test:
        print('the test samples save_index ', save_index, ' is not equal to the whole test set ', self.num_test)

    print('Tested on : ', pred_gaze_all.shape[0], ' samples')
    np.savetxt('within_eva_results.txt', pred_gaze_all, delimiter=',')

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