import pathlib
from typing import Callable, Tuple
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, IterableDataset
from typing import List
import os
import json
import random

class OnePersonDataset(Dataset):
    def __init__(self, person_id_str: str, dataset_path: pathlib.Path,
                 transform: Callable, load_model: str):
        self.person_id_str = person_id_str
        self.dataset_path = dataset_path
        self.transform = transform
        self.load_model = load_model

    def __getitem__(
            self,
            index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        initial_image = None
        initial_gaze = None
        
        with h5py.File(self.dataset_path, 'r') as f:
            # print(self.person_id_str)
            self.person_id_str = int(self.person_id_str[1:])
            self.person_id_str = f'{self.person_id_str:02}'
            image = f.get(f'{self.person_id_str}/image/{index:04}')[()]
            gaze = f.get(f'{self.person_id_str}/gaze/{index:04}')[()]
            
            
#             image_path = f'{self.person_id_str}/image/{index:04}'
#             gaze_path = f'{self.person_id_str}/gaze/{index:04}'

#             image = f.get(image_path)
#             if image is None:
#                 if initial_image is not None:
#                     image = initial_image
#                     print(f"Replaced image for index {index}")
#             else:
#                 image = image[()]
#                 if initial_image is None:
#                     initial_image = image

#             gaze = f.get(gaze_path)
#             if gaze is None:
#                 if initial_gaze is not None:
#                     gaze = initial_gaze
#                     print(f"Replaced gaze for index {index}")
#             else:
#                 gaze = gaze[()]
#                 if initial_gaze is None:
#                     initial_gaze = gaze 
                    
#                     '''上面被修改过'''
            if self.load_model == 'load_multi_region':
                left_eye = f.get(f'{self.person_id_str}/left/{index:04}')[()]
                right_eye = f.get(f'{self.person_id_str}/right/{index:04}')[()]
                
        image=np.transpose(image, (1,2,0)) #(C,H,W) -> (H,W,C)
        #BELOW IS TO CROP IMAGE,NOW WE ARE USING RESIZE TO CHANGE SIZE
        # '''from 448*448*3 to 224*224*3'''       
        # start_x = (image.shape[1] - 224) // 2
        # start_y = (image.shape[0] - 224) // 2
        # image = image[start_y:start_y + 224, start_x:start_x + 224]


        image = image[:, :, [2, 1, 0]]  #from BGR to RGB
        image = self.transform(image)
        gaze = torch.from_numpy(gaze)
        gaze = gaze[:2]
        if self.load_model == 'load_single_face':
            images = {"face": image}
        # if self.load_model == 'load_multi_region':
        #     left_eye = self.transform(left_eye)
        #     right_eye = self.transform(right_eye)
        #     images = {"face": image, "left_eye": left_eye, "right_eye": right_eye}
        return images, gaze  #, left_eye, right_eye

    def __len__(self) -> int:
        return 3000


class Gaze360IterableDataset(IterableDataset):
    def __init__(self, full_path, transform: Callable, load_model: str):
        self.transform = transform
        self.load_model = load_model
        self.full_path = full_path
        with h5py.File(self.full_path, 'r') as f:
            self.length = len(f['face'])

    def __iter__(self):
        with h5py.File(self.full_path, 'r') as f:
            image_data = f['face']
            gaze_data = f['gaze']
            for i in range(len(image_data)):
                image = image_data[i]
                gaze = gaze_data[i]
                
                image = image[:, :, [2, 1, 0]]  #from BGR to RGB
                # Apply transformations
                image = self.transform(image)
                gaze = torch.from_numpy(gaze)
                
                if self.load_model == 'load_single_face':
                    images = {"face": image}
                elif self.load_model == 'load_multi_region':
                    # Replace with actual left_eye and right_eye extraction logic
                    left_eye = self.transform(image)  # Placeholder
                    right_eye = self.transform(image)  # Placeholder
                    images = {"face": image, "left_eye": left_eye, "right_eye": right_eye}
                
                yield images, gaze
                
    def __len__(self):
        return self.length



class XGazeDataset(Dataset):
    def __init__(self, dataset_path: str, keys_to_use: List[str] = None, sub_folder='', transform=None, is_shuffle=True,
                 index_file=None, is_load_label=True):
        self.path = dataset_path
        self.hdfs = {} # 用于存储打开的HDF5文件
        self.sub_folder = sub_folder
        self.is_load_label = is_load_label

        # assert len(set(keys_to_use) - set(all_keys)) == 0
        # Select keys
        # TODO: select only people with sufficient entries?
        # 确保所选的键值（self.selected_keys）在所有键值(all_keys)中均存在，否则会触发断言错误。
        self.selected_keys = [k for k in keys_to_use]
        assert len(self.selected_keys) > 0

        for num_i in range(0, len(self.selected_keys)):
            file_path = os.path.join(self.path, self.sub_folder, self.selected_keys[num_i])
            self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
            # print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))
            assert self.hdfs[num_i].swmr_mode # 确保HDF5文件处于单写多读模式

        # Construct mapping from full-data index to key and person-specific index
        if index_file is None:
            self.idx_to_kv = []
            for num_i in range(0, len(self.selected_keys)):
                n = self.hdfs[num_i]["face_patch"].shape[0] # 获取face_patch数据集的行数?
                self.idx_to_kv += [(num_i, i) for i in range(n)] # 将每个人的face_patch数据集的行数与人的索引对应起来?
        else:
            print('load the file: ', index_file)
            self.idx_to_kv = np.loadtxt(index_file, dtype=np.int)

        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

        if is_shuffle:
            random.shuffle(self.idx_to_kv)  # random the order to stable the training

        self.hdf = None # 用于存储当前打开的HDF5文件
        self.transform = transform

    def __len__(self):  # 返回数据集的长度，即数据样本的数量。
        return len(self.idx_to_kv)

    def __del__(self):  # 析构函数
        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

    def __getitem__(self, idx): # 根据索引idx返回数据集中的一个样本
        key, idx = self.idx_to_kv[idx]

        self.hdf = h5py.File(os.path.join(self.path, self.sub_folder, self.selected_keys[key]), 'r', swmr=True)
        assert self.hdf.swmr_mode

        # Get face image
        image = self.hdf['face_patch'][idx, :]
        image = image[:, :, [2, 1, 0]]  # from BGR to RGB
        image = self.transform(image)

        # Get labels
        if self.is_load_label:
            gaze_label = self.hdf['face_gaze'][idx, :]
            gaze_label = gaze_label.astype('float')
            return image, gaze_label
        else:
            return image