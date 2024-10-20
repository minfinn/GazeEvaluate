import pathlib
from typing import Callable, Tuple
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, IterableDataset


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
                
        image=np.transpose(image, (1,2,0))
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
