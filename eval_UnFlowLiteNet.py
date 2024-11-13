# -*- coding: utf-8 -*-
"""
UnLiteFlowNet-PIV with Custom Data
python mainPIV.py --test
python mainPIV.py --train
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from src.model.models import *
from src.train.train_functions import *
import matplotlib.pyplot as plt

#

def plot_velocity_comparison(u_true, v_true, u_pred, v_pred, coord1=(60, 60), coord2=(100, 100)):
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
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Velocity u component plot
    axs[0].plot(u_true[:, coord1[0], coord1[1]], label=f"True (u) {coord1}")
    axs[0].plot(u_pred[:, coord1[0], coord1[1]], label=f"Predicted (u) {coord1}")
    #axs[0].plot(u_true[:, coord2[0], coord2[1]], label=f"True (u) {coord2}")
    #axs[0].plot(u_pred[:, coord2[0], coord2[1]], '--', label=f"Predicted (u) {coord2}")
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('u (x, y) m/s')
    axs[0].set_title(f'Velocity u(x,y,t) at coordinates {coord1}')
    axs[0].grid(True)
    axs[0].legend()

    # Velocity v component plot
    axs[1].plot(v_true[:, coord1[0], coord1[1]], label=f"True (v) {coord1}")
    axs[1].plot(v_pred[:, coord1[0], coord1[1]], label=f"Predicted (v) {coord1}")
    #axs[1].plot(v_true[:, coord2[0], coord2[1]], label=f"True (v) {coord2}")
    #axs[1].plot(v_pred[:, coord2[0], coord2[1]], '--', label=f"Predicted (v) {coord2}")
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('v (x, y) m/s')
    axs[1].set_title(f'Velocity v(x,y,t) at coordinates {coord1}')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("Velocity_uv_PIV_HDR_comparison_PRED_0.png")
    plt.show()

# Load your data
inputs_data = np.load("../datas/clean_data/clean_data/noisy_images_data_H017.npy")
targets_data = np.load("../datas/clean_data/clean_data/data_ground_truth_H017.npy")

# Hyperparameters
lr = 1e-4
batch_size = 4
test_batch_size = 4
n_epochs = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Custom Dataset Class
class CustomFlowDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        #self.inputs = torch.from_numpy(inputs).float()
        #self.targets = torch.from_numpy(targets).float()
        max_pixel_value = max(inputs.max(), targets.max())
        self.max_pixel_value = max_pixel_value
        
        self.inputs = torch.from_numpy(inputs.astype(np.float32)) / max_pixel_value #torch.from_numpy(inputs.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = self.inputs[idx]  # Shape (120, 120)
        target_data = self.targets[idx]  # Shape (2, 120, 120)
        return input_data, target_data


def test_train():
    # Prepare dataset
    dataset = CustomFlowDataset(inputs_data, targets_data)
    total_images = len(dataset)

    # Calculate train, validation, and test sizes as multiples of 8
    train_size = 400 #int(0.8 * total_images) // 8 * 8  # Closest multiple of 8
    val_size = 40 #int(0.1 * total_images) // 8 * 8     # Closest multiple of 8
    test_size = int(total_images - train_size - val_size) # Remainder for test set

    # Ensure data is split sequentially
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, total_images))

    # Load the network model
    model = Network().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8, eps=1e-6, amsgrad=True)
    # Train model
    model_trained = train_model(model, train_dataset, val_dataset, test_dataset, batch_size, test_batch_size, lr, n_epochs, optimizer)

    
    test_code = "31_Octobre_2024"
  # Save the trained model
    model_save_path = f"./models/model_saved_{test_code}.pt"
    #model_save_path = ./models/model_saved_31_Octobre_2024.pt
    torch.save({'model_state_dict': model_trained.state_dict()}, model_save_path)
    print(f"Model saved to {model_save_path}")
    return model_trained


import numpy as np
import torch.nn.functional as F

def test_estimate_all():
    # Prepare test dataset
    dataset = CustomFlowDataset(inputs_data, targets_data)
    total_images = len(dataset)

    # Calculate train, validation, and test sizes as multiples of 8
    train_size = 300 #400 #int(0.8 * total_images) // 8 * 8  # Closest multiple of 8
    val_size = 8 #40 #int(0.1 * total_images) // 8 * 8     # Closest multiple of 8
    test_size = int(total_images - train_size - val_size) # Remainder for test set

    # Ensure data is split sequentially
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, total_images))

    # Load pretrained model
    model_save_name = 'UnsupervisedLiteFlowNet_pretrained.pt'
    PATH = "./models/UnsupervisedLiteFlowNet_pretrained.pt"
    unliteflownet = Network().to(device)
    unliteflownet.load_state_dict(torch.load(PATH)['model_state_dict'])
    unliteflownet.eval()

    # Initialize lists to store all predictions
    u_preds, v_preds = [], []
    u_trues, v_trues = [], []
    print(len(test_dataset))
    # Process each sample in the test dataset
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            input_data, label_data = test_dataset[idx]
            h_origin, w_origin = input_data.shape[-2], input_data.shape[-1]

            # Prepare input frames
            tensorFirst = input_data[0].unsqueeze(0).unsqueeze(0).expand(1, 1, 120, 120).to(device)
            tensorSecond = input_data[1].unsqueeze(0).unsqueeze(0).expand(1, 1, 120, 120).to(device)
            x1 = F.interpolate(tensorFirst, size=(256, 256), mode='bilinear', align_corners=False)
            x2 = F.interpolate(tensorSecond, size=(256, 256), mode='bilinear', align_corners=False)

            # Prediction
            y_pre = estimate(x1, x2, unliteflownet, train=False)
            y_pre = F.interpolate(y_pre, size=(h_origin, w_origin), mode='bilinear', align_corners=False)

            # Extract and store predicted u and v components
            u_pred = y_pre[0][0].cpu().numpy()
            v_pred = y_pre[0][1].cpu().numpy()
            u_preds.append(u_pred)
            v_preds.append(v_pred)

            # Extract and store true u and v components
            u_true = label_data[0].cpu().numpy()
            v_true = label_data[1].cpu().numpy()
            u_trues.append(u_true)
            v_trues.append(v_true)

    # Convert lists to numpy arrays for easier handling
    u_preds = np.array(u_preds)
    v_preds = np.array(v_preds)
    u_trues = np.array(u_trues)
    v_trues = np.array(v_trues)

    # Plot comparison for the entire dataset at coordinates (60, 60) and (100, 100)
    plot_velocity_comparison(u_trues, v_trues, u_preds, v_preds, coord1=(60, 60), coord2=(100, 100))


def test_estimate():
    # Prepare test dataset
    test_dataset = CustomFlowDataset(inputs_data, targets_data)

    # Load pretrained model
    model_save_name = 'UnsupervisedLiteFlowNet_pretrained.pt'
    PATH = f"./models/{model_save_name}"
    #PATH = "./models/model_saved_31_Octobre_2024.pt"
    unliteflownet = Network().to(device)
    unliteflownet.load_state_dict(torch.load(PATH)['model_state_dict'])
    unliteflownet.eval()
    print(len(test_dataset))
    # Select random sample from test dataset
    number = 0 #random.randint(0, len(test_dataset) - 1)
    input_data, label_data = test_dataset[number]
    h_origin, w_origin = input_data.shape[-2], input_data.shape[-1]
    #number = random.randint(0, len(test_dataset) - 1)
    #input_data, label_data = test_dataset[number]
    
    # Split input_data into two frames
    #tensorFirst = input_data[0].unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dimensions
    #tensorSecond = input_data[1].unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dimensions
    tensorFirst = input_data[0].unsqueeze(0).unsqueeze(0).expand(1, 1, 120, 120).to(device)
    tensorSecond = input_data[1].unsqueeze(0).unsqueeze(0).expand(1, 1, 120, 120).to(device)
    
    x1 = F.interpolate(tensorFirst, size=(256, 256), mode='bilinear', align_corners=False)
    x2 = F.interpolate(tensorSecond, size=(256, 256), mode='bilinear', align_corners=False)

    print("Shape of tensorFirst:", tensorFirst.shape)
    print("Shape of tensorSecond:", tensorSecond.shape)



    # Prediction
    with torch.no_grad():
        #pred_flow = model(tensorFirst, tensorSecond) #model(input_data)
        y_pre = estimate(x1, x2, unliteflownet, train=False)
    y_pre = F.interpolate(y_pre, size=(h_origin, w_origin), mode='bilinear', align_corners=False)
    print(y_pre.shape)
    # Adjust for resizing ratios
    resize_ratio_u = h_origin / 256
    resize_ratio_v = w_origin / 256
    u_pred = y_pre[0][0].cpu().numpy()* resize_ratio_u
    v_pred = y_pre[0][1].cpu().numpy()* resize_ratio_v
    #pred_flow = F.interpolate(pred_flow, size=(120, 120), mode='bilinear', align_corners=False)
    # Visualization
    fig, axarr = plt.subplots(1, 2, figsize=(16, 8))

    # Predicted flow visualization
    #u_pred, v_pred = pred_flow[0, 0].cpu().numpy(), pred_flow[0, 1].cpu().numpy()
    flow_magnitude_pred = np.sqrt(u_pred**2 + v_pred**2)
    im = axarr[0].imshow(flow_magnitude_pred, cmap='inferno')
    axarr[0].title.set_text("Predicted Flow Magnitude")
    fig.colorbar(im, ax=axarr[0], fraction=0.046, pad=0.04)

    # Ground truth flow visualization
    u_true, v_true = label_data[0].cpu().numpy(), label_data[1].cpu().numpy()
    flow_magnitude_true = np.sqrt(u_true**2 + v_true**2)
    im = axarr[1].imshow(flow_magnitude_true, cmap='inferno')
    axarr[1].title.set_text("Ground Truth Flow Magnitude")
    fig.colorbar(im, ax=axarr[1], fraction=0.046, pad=0.04)

    plt.savefig(f"Flow_estimate_using_UnLite_Time{number}.png")
    plot_velocity_comparison(u_true, v_true, u_pred, v_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and test')
    parser.add_argument('--train', action='store_true', help='train the model')
    parser.add_argument('--test', action='store_true', help='test the model')

    args = parser.parse_args()
    if args.train:
        test_train()
    if args.test:
        #test_estimate()
        test_estimate_all()
