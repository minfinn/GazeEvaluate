import pathlib
from typing import List, Union
import json
import torch
import yacs.config
import os
import random
from torch.utils.data import Dataset
from ..transforms import create_transform
from ..types import GazeEstimationMethod
from .mpiifacegaze import OnePersonDataset, Gaze360IterableDataset, XGazeDataset, ColumbiaIterableDataset, EVEIterableDataset


def create_dataset(config: yacs.config.CfgNode,
                   is_train: bool = True) -> Union[List[Dataset], Dataset]:

    dataset_dir = pathlib.Path(config.dataset.dataset_dir)

    if config.dataset.name == 'MPII':    
        

        assert dataset_dir.exists()
        assert config.train.test_id in range(-1, 15)
        assert config.test.test_id in range(15)
        person_ids = [f'p{index:02}' for index in range(15)]

        transform = create_transform(config)

        if config.model.name == 'face_res50':
            load_mode = 'load_single_face'
        elif config.model.name == 'multi_region_res50':
            load_mode = 'load_multi_region'
        elif config.model.name == 'multi_region_res50_share_eyenet':
            load_mode = 'load_multi_region'
        else:
            raise Exception("Please enter a correct model name or choose a correct load mode for your model (load_single_face or load_multi_region).")



        if is_train:
            if config.train.test_id == -1:
                train_dataset = torch.utils.data.ConcatDataset([
                    OnePersonDataset(person_id, dataset_dir, transform, load_mode)
                    for person_id in person_ids
                ])
                print('load oneperson successfully')
                assert len(train_dataset) == 45000
            else:
                test_person_id = person_ids[config.train.test_id]
                train_dataset = torch.utils.data.ConcatDataset([
                    OnePersonDataset(person_id, dataset_dir, transform, load_mode)
                    for person_id in person_ids if person_id != test_person_id
                ])
                assert len(train_dataset) == 42000

            val_ratio = config.train.val_ratio
            assert val_ratio < 1
            val_num = int(len(train_dataset) * val_ratio)
            train_num = len(train_dataset) - val_num
            lengths = [train_num, val_num]
            return torch.utils.data.dataset.random_split(train_dataset, lengths)
        else:
            test_person_id = person_ids[config.test.test_id]
            test_dataset = OnePersonDataset(test_person_id, dataset_dir, transform, load_mode)
            assert len(test_dataset) == 3000
            return test_dataset
    elif config.dataset.name == 'GAZE360':
        # train_path=dataset_dir / 'train'
        # transform = create_transform(config)
        # load_mode = 'load_single_face'
    #     train_dataset = torch.utils.data.ConcatDataset([
    #                 Gaze360IterableDataset(train_path / file.name, transform, load_mode)
    # for file in train_path.iterdir() if file.suffix == '.h5'
    #             ])
        # print('load onedir successfully')
        
        # val_path=dataset_dir / 'val'
        # transform = create_transform(config)
    #     val_dataset = torch.utils.data.ConcatDataset([
    #                 Gaze360IterableDataset(val_path / file.name, transform, load_mode)
    # for file in val_path.iterdir() if file.suffix == '.h5'
    #             ])
    
        # return train_dataset,val_dataset
        train_path = dataset_dir / 'train'
        val_path = dataset_dir / 'val'
        transform = create_transform(config)
        load_mode = 'load_single_face'

        train_files = [file for file in train_path.iterdir() if file.suffix == '.h5']
        val_files = [file for file in val_path.iterdir() if file.suffix == '.h5']

        # 创建 DataLoader
        train_dataset = [Gaze360IterableDataset(file, transform, load_mode) for file in train_files]
        val_dataset = [Gaze360IterableDataset(file, transform, load_mode) for file in val_files]
        
        return train_dataset,val_dataset     


def create_testset(config: yacs.config.CfgNode,
                   is_train: bool = True) -> Union[List[Dataset], Dataset]:
    
    
    dataset_dir = pathlib.Path(config.dataset.data_dir)

    if config.dataset.name == 'MPII':
        assert dataset_dir.exists()
        
        # assert config.train.test_id in range(-1, 15)
        assert config.test.test_id in range(15)


        person_ids = [f'p{index:02}' for index in range(15)]
        # print(person_ids)
        assert dataset_dir.exists()
        transform = create_transform(config)

        if config.model.name == 'face_res50':
            load_mode = 'load_single_face'
        elif config.model.name == 'multi_region_res50':
            load_mode = 'load_multi_region'
        elif config.model.name == 'multi_region_res50_share_eyenet':
            load_mode = 'load_multi_region'
        else:
            raise Exception("Please enter a correct model name or choose a correct load mode for your model (load_single_face or load_multi_region).")


        test_person_id = person_ids[config.test.test_id]
        test_dataset = OnePersonDataset(test_person_id, dataset_dir, transform, load_mode)
        assert len(test_dataset) == 3000
        return test_dataset
    

    elif config.dataset.name == 'GAZE360':
        test_path = dataset_dir / 'test'
        assert test_path.exists()
        transform = create_transform(config)

        if config.model.name == 'face_res50':
            load_mode = 'load_single_face'
        elif config.model.name == 'multi_region_res50':
            load_mode = 'load_multi_region'
        elif config.model.name == 'multi_region_res50_share_eyenet':
            load_mode = 'load_multi_region'
        else:
            raise Exception("Please enter a correct model name or choose a correct load mode for your model (load_single_face or load_multi_region).")

        test_files = [file for file in test_path.iterdir() if file.suffix == '.h5']
        
        test_dataset = [Gaze360IterableDataset(file, transform, load_mode) for file in test_files]


    elif config.dataset.name == 'XGAZE':
        '''
        test_dataset creation of XGAZE base on original code offered by ETH-XGaze
        '''
        assert dataset_dir.exists()
        transform = create_transform(config)
        refer_list_file = os.path.join(dataset_dir, 'train_test_split.json')

        print('load the test file list from: ', refer_list_file)

        with open(refer_list_file, 'r') as f:
            datastore = json.load(f)

        # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
        # train set: the training set includes 80 participants data
        # test set: the test set for cross-dataset and within-dataset evaluations
        # test_person_specific: evaluation subset for the person specific setting
        sub_folder_use = 'test'
        test_dataset = XGazeDataset(dataset_path=dataset_dir, keys_to_use=datastore[sub_folder_use], sub_folder=sub_folder_use,
                            transform=transform, is_shuffle=True, is_load_label=False)

        return test_dataset
    
    elif config.dataset.name == 'ColumbiaGaze':
        
        test_path = dataset_dir / 'test'
        assert test_path.exists()
        print('load onedir successfully')
        transform = create_transform(config)

        if config.model.name == 'face_res50':
            load_mode = 'load_single_face'
        elif config.model.name == 'multi_region_res50':
            load_mode = 'load_multi_region'
        elif config.model.name == 'multi_region_res50_share_eyenet':
            load_mode = 'load_multi_region'
        else:
            raise Exception("Please enter a correct model name or choose a correct load mode for your model (load_single_face or load_multi_region).")

        test_files = [file for file in test_path.iterdir() if file.suffix == '.h5'] #get .h5 test file
        test_num = len(test_files)

        # random.shuffle(train_val_files)

        # train_files = train_val_files[:train_num]
        # val_files = train_val_files[train_num:]

        test_dataset = [ColumbiaIterableDataset(file, transform, load_mode) for file in test_files]
            
        return test_dataset
    
    elif config.dataset.name == 'EVE':

        test_paths = [dataset_dir / f'test{i:02d}' for i in range(1,11)]#participant
        for test_path in test_paths:
            assert test_path.exists()
        print('load dir successfully')


        if config.model.name == 'face_res50':
            load_mode = 'load_single_face'
        elif config.model.name == 'multi_region_res50':
            load_mode = 'load_multi_region'
        elif config.model.name == 'multi_region_res50_share_eyenet':
            load_mode = 'load_multi_region'
        else:
            raise Exception("Please enter a correct model name or choose a correct load mode for your model (load_single_face or load_multi_region).")
        test_sub_paths = []
        for test_path in test_paths:
            subfolders = [folder for folder in test_path.iterdir() if folder.is_dir()]
            test_sub_paths.extend(subfolders)  #subfolder


        transform = create_transform(config)
        test_files = []
        
        for test_sub_path in test_sub_paths:
            for file in test_sub_path.iterdir(): #cam
                if file.suffix == '.h5':
                    test_files.append(file)
        
        test_dataset = [EVEIterableDataset(file, transform, load_mode) for file in test_files]
        
        return test_dataset