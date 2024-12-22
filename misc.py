# -*- coding: utf-8 -*-
"""
UnLiteFlowNet-PIV

"""
import argparse
import numpy as np
import os


def test_train(args=None):
    # Load the data
    img_dim = (256, 256)

    inputs_data = np.load(args.input_path)
    targets_data = np.load(args.target_path)
    
    if args.case == 0:
        inputs_data = inputs_data[:, :120, :120, :]
        targets_data = targets_data[:, :120, :120, :]
        
        input_file_name = args.input_path.split('/')[-1]
        input_name = input_file_name.split('.')[0]
        input_save_path = args.input_path.replace(input_name, f'{input_name}_Cropped_120x120')
        np.save(input_save_path, inputs_data)
        
        input_file_name = args.target_path.split('/')[-1]
        input_name = input_file_name.split('.')[0]
        input_save_path = args.target_path.replace(input_name, f'{input_name}_Cropped_120x120')
        
        np.save(input_save_path, targets_data)
        
    if args.case == 1:
        B, H, W = inputs_data.shape
        input_padded = np.zeros((B, 256, 256))
        input_padded[:, :120, :120] = inputs_data
        
        input_file_name = args.input_path.split('/')[-1]
        input_name = input_file_name.split('.')[0]
        input_save_path = args.input_path.replace(input_name, f'{input_name}_Padded_120x120')
        np.save(input_save_path, input_padded)
        
        B, T, H, W = targets_data.shape
        target_padded = np.zeros((B, T, 256, 256))
        target_padded[:, :, :120, :120] = targets_data
        
        input_file_name = args.target_path.split('/')[-1]
        input_name = input_file_name.split('.')[0]
        input_save_path = args.target_path.replace(input_name, f'{input_name}_Padded_120x120')
        
        np.save(input_save_path, target_padded)
    print("Done !")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--input_path', type=str, help='Path to the input data')
    parser.add_argument('--target_path', type=str, help='Path to the target data')
    parser.add_argument('--case', type=int, help='Path to the data')

    args = parser.parse_args()
    model = test_train(args)
