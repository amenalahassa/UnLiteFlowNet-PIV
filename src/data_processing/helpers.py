# Helpers functions
from src.data_processing.local_dataset import CustomFlowDataset
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import numpy as np
import cv2  # Utilisé pour la redimension via interpolation (OpenCV)

def compute_aee_per_x_pixels(aee, image_width, x_pixels=100):
    """
    Compute the Averaged Endpoint Error (AEE) per X pixels.

    Args:
        aee (float): The original AEE value (averaged endpoint error).
        image_width (int): The width of the image in pixels.
        x_pixels (int): The normalization factor, default is 100 pixels.

    Returns:
        float: AEE normalized per X pixels.
    """
    scaling_factor = x_pixels / image_width
    aee_per_x_pixels = aee * scaling_factor
    return aee_per_x_pixels

def compute_rmse(u_pred, u_true, v_pred, v_true):
    rmse = np.sqrt(np.mean((u_pred - u_true)**2 + (v_pred - v_true)**2))
    return rmse

def compute_aee(u_pred, u_true, v_pred, v_true):
    ee = np.sqrt((u_pred - u_true)**2 + (v_pred - v_true)**2)
    aee = np.mean(ee)
    return aee


def create_consecutive_pairs_v1(inputs_data, output_size=(256, 256)):
    
    if len(inputs_data.shape) < 4:
        # Crée des paires consécutives (img1, img2), (img2, img3), ..., (img497, img498)
        inputs_pairs = [(inputs_data[i], inputs_data[i + 1]) for i in range(len(inputs_data) - 1)]
        inputs_pairs = np.array(inputs_pairs)  # (497, 2, 1, 120, 120)
    else:
        inputs_pairs = inputs_data
        
    # print(inputs_data.shape, inputs_pairs.shape)
    if inputs_pairs.shape[-1] == 2:
        inputs_pairs = inputs_pairs.transpose(0, 3, 1, 2)
        
    
    B, T, H, W = inputs_pairs.shape
    if output_size != None and output_size != (H, W):
        # Redimensionner les paires d'entrées à (256, 256)
        inputs_resized = np.array([
            [
                cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR) 
                for img in pair
            ]
            for pair in inputs_pairs
        ])

    else:
        return inputs_pairs
    
    # print(inputs_resized.shape, inputs_pairs.shape)
    # return inputs_resized[:, np.newaxis, :, :, :], targets_resized   # Résultat avec la taille (497, 2, 256, 256)
    return inputs_resized   # Résultat avec la taille (497, 2, 256, 256)


def create_consecutive_pairs_v2(inputs_data, targets_data, output_size=(256, 256)):
    
    if len(inputs_data.shape) < 4:
        # Crée des paires consécutives (img1, img2), (img2, img3), ..., (img497, img498)
        inputs_pairs = [(inputs_data[i], inputs_data[i + 1]) for i in range(len(inputs_data) - 1)]
    
        # Convertir en numpy array et ajouter une dimension pour obtenir (497, 2, 120, 120)
        # inputs_pairs = np.array(inputs_pairs)[:, np.newaxis, :, :, :]  # (497, 2, 1, 120, 120)
        inputs_pairs = np.array(inputs_pairs)  # (497, 2, 1, 120, 120)
    else:
        inputs_pairs = inputs_data
        
    # print(inputs_pairs.shape)
    if inputs_pairs.shape[-1] == 2:
        inputs_pairs = inputs_pairs.transpose(0, 3, 1, 2)
        
    if targets_data.shape[-1] == 2:
        targets_data = targets_data.transpose(0, 3, 1, 2)
        
    B, T, H, W = inputs_pairs.shape
    if output_size != None and output_size != (H, W):
        # Redimensionner les paires d'entrées à (256, 256)
        inputs_resized = np.array([
            [
                cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR) 
                for img in pair[0]
            ]
            for pair in inputs_pairs
        ])

        # Redimensionner les cibles (targets_data[1:]) à (256, 256)
        targets_resized = np.array([
            [
                cv2.resize(targets_data[i, j], output_size, interpolation=cv2.INTER_LINEAR)
                for j in range(targets_data.shape[1])
            ]
            for i in range(1, len(targets_data))
        ])  # (48, 2, 256, 256)
    else:
        return inputs_pairs, targets_data
    
    # return inputs_resized[:, np.newaxis, :, :, :], targets_resized   # Résultat avec la taille (497, 2, 256, 256)
    return inputs_resized, targets_resized   # Résultat avec la taille (497, 2, 256, 256)

def buildDataset(inputs_data, targets_data, test_size=0.2, eval_size=0.1, img_size=None, use_denoising="gaussian", denoising_kwargs=None):
    # Prepare test dataset
    inputs_data, targets_data = create_consecutive_pairs_v2(inputs_data, targets_data, output_size=None)
    total_images = len(inputs_data)

    test_size = int(test_size * total_images)
    eval_size = int(eval_size * total_images)
    train_size = total_images - test_size

    train_dataset = CustomFlowDataset(inputs_data[:train_size], targets_data[:train_size], img_size=img_size, use_denoising=use_denoising, denoising_kwargs=denoising_kwargs)
    test_dataset = CustomFlowDataset(inputs_data[train_size:], targets_data[train_size:], img_size=img_size, use_denoising=use_denoising, denoising_kwargs=denoising_kwargs)

    train_size = train_size - eval_size

    train_dataset, validate_dataset = random_split(train_dataset, [train_size, eval_size])

    return train_dataset, validate_dataset, test_dataset

def plot_velocity_comparison(u_true, v_true, u_pred, v_pred, coord = (60, 60), save_path=None):
    """
    Plots a comparison of true vs. predicted velocity components at specific coordinates.

    Args:
        u_true (np.array): True u velocity component, shape (time, height, width).
        v_true (np.array): True v velocity component, shape (time, height, width).
        u_pred (np.array): Predicted u velocity component, shape (time, height, width).
        v_pred (np.array): Predicted v velocity component, shape (time, height, width).
        coord1 (tuple): First coordinate (x, y) for comparison.
        coord2 (tuple): Second coordinate (x, y) for comparison.
    """
    coord1 = coord
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Velocity u component plot
    axs[0].plot(u_true[:, coord1[0], coord1[1]], label=f"True (u) {coord1}")
    axs[0].plot(u_pred[:, coord1[0], coord1[1]], label=f"Predicted (u) {coord1}")
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('u (x, y) m/s')
    axs[0].set_title(f'Velocity u(x,y,t) at coordinates {coord1}')
    axs[0].grid(True)
    axs[0].legend()

    # Velocity v component plot
    axs[1].plot(v_true[:, coord1[0], coord1[1]], label=f"True (v) {coord1}")
    axs[1].plot(v_pred[:, coord1[0], coord1[1]], label=f"Predicted (v) {coord1}")
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('v (x, y) m/s')
    axs[1].set_title(f'Velocity v(x,y,t) at coordinates {coord1}')
    axs[1].grid(True)
    axs[1].legend()

    # plt.tight_layout()
    plt.savefig(save_path, format="png")
    plt.show()
    plt.close()
