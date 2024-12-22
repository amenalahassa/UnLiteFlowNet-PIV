import torch
import numpy as np
from src.data_processing.sp_functions import denoise_piv_images
import numpy as np
import cv2 

class CustomFlowDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets, use_denoising="gaussian", denoising_kwargs=None, img_size=None, normalize = False):
        # Todo: ask why we are dividing by the max_pixel_value
        if normalize:
            max_pixel_value = max(inputs.max(), targets.max())
            self.max_pixel_value = max_pixel_value
        else:
            max_pixel_value = 1

        self.inputs = torch.from_numpy(inputs.astype(np.float32)) / max_pixel_value #torch.from_numpy(inputs.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))
        # print(self.inputs.shape)

        self.use_denoising = use_denoising
        self.denoising_kwargs = denoising_kwargs
        self.img_size = img_size

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = self.inputs[idx]  # Shape (120, 120)
        target_data = self.targets[idx]  # Shape (2, 120, 120)
        T, H, W = input_data.shape
        

        # Apply denoising to input data
        if self.use_denoising != 'none':
            input_data = denoise_piv_images(input_data, self.use_denoising, **self.denoising_kwargs)
            
            
        if self.img_size != None and self.img_size != (H, W):
            
            input_data = np.array([
                cv2.resize(img.numpy(), self.img_size, interpolation=cv2.INTER_LINEAR)
                for img in input_data
            ])
            target_data = np.array([
                cv2.resize(img.numpy(), self.img_size, interpolation=cv2.INTER_LINEAR)
                for img in target_data
            ])
            
        return input_data, target_data  
