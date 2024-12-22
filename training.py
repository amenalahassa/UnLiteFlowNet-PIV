# -*- coding: utf-8 -*-
"""
UnLiteFlowNet-PIV

"""
import argparse
from src.model.models import *
from src.train.train_functions import *
from src.data_processing.local_dataset import *
from src.data_processing.helpers import *
import wandb
import os


def test_train(args=None):
    # Load the data
    use_whole_data = False
    img_dim = (256, 256)
    
    
    if args.data_path:
        data_path = args.data_path
        files = os.listdir(data_path)
        inputs_data = []
        targets_data = []
        # print(files)
        for file in reversed(files):
            if 'ground_truth' in file and file.endswith('.npy'):
                truth = create_consecutive_pairs_v1(np.load(os.path.join(data_path, file)), output_size=img_dim)
                # print('ground_truth', truth.shape, truth.max(), truth.min())
                targets_data.append(truth)
            elif 'noisy' in file and file.endswith('.npy'):
                noise = create_consecutive_pairs_v1(np.load(os.path.join(data_path, file)), output_size=img_dim)
                # print('noisy', noise.shape, noise.max(), noise.min())
                
                if args.normalise == 1:
                    noise = noise / noise.max() 
                
                inputs_data.append(noise)
                
        # Concatenate the data
        print([i.shape[0] for i in inputs_data])
        inputs_data = np.concatenate(inputs_data, axis=0)
        targets_data = np.concatenate(targets_data, axis=0)
        use_whole_data = True
    else:
        inputs_data = np.load(args.input_path)
        
        if args.normalise == 1:
            inputs_data = inputs_data / inputs_data.max() 
                
        targets_data = np.load(args.target_path)
        # print(inputs_data.shape, targets_data.shape)
    
    # Prepare the dataset
    test_size = 0.2
    eval_size = 0.1

    # Set denoise method
    use_denoising = args.use_denoising
    denoising_kwargs = None
    if use_denoising == 'gaussian':
        denoising_kwargs = {'sigma': 1.5}
    elif use_denoising == 'median':
        denoising_kwargs = {'size': 5}
    else:
        denoising_kwargs = {}

    # Build the dataset
    train_dataset, validate_dataset, test_dataset = buildDataset(inputs_data, targets_data, img_size=img_dim, test_size=test_size, eval_size=eval_size, use_denoising=use_denoising, denoising_kwargs=denoising_kwargs)

    # Set hyperparameters
    lr = 1e-4
    weight_decay = 1e-5
    eps = 1e-3
    batch_size = 8
    test_batch_size = 8 * 2
    n_epochs = 100
    new_train = new_train = args.new_train


    # Load the network model
    model = Network().to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay,
                                 eps=eps,
                                 amsgrad=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    if args.wandb:
        wandb.login(key=args.wandb)
        config = {
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "eps": eps,
            "batch_size": batch_size,
            "test_batch_size": test_batch_size,
            "n_epochs": n_epochs,
            "checkpoint": args.model_save_name,
            "use_denoising": use_denoising,
            "denoising_kwargs": denoising_kwargs,
            'data_path': args.data_path,
            'input_path': args.input_path,
            'target_path': args.target_path,
            'use_whole_data': use_whole_data,
            'use_pretrained': args.use_pretrained,
            'train_size': len(train_dataset),
            'test_size': len(test_dataset),
            'eval_size': len(validate_dataset),
            'normalise': args.normalise,
        }
        wandb.init(project=args.experiment_name, config=config)
        wandb.watch(model, log="all")

    if new_train:
        
        if args.use_pretrained:
            model_save_name = args.model_save_name
            PATH = F"./models/{model_save_name}"
            checkpoint = torch.load(PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            
        # New train
        model_trained = train_model(model, train_dataset, validate_dataset,
                                    test_dataset, batch_size, test_batch_size,
                                    lr, n_epochs, optimizer, args=args)
    else:
        model_save_name = args.model_save_name
        PATH = F"./models/{model_save_name}"
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

        model_trained = train_model(model,
                                    train_dataset,
                                    validate_dataset,
                                    test_dataset,
                                    batch_size,
                                    test_batch_size,
                                    lr,
                                    n_epochs,
                                    optimizer,
                                    epoch_trained=epoch + 1, args=args)
    return model_trained


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--input_path', type=str, help='Path to the input data')
    parser.add_argument('--target_path', type=str, help='Path to the target data')
    parser.add_argument('--data_path', type=str, help='Path to the data')
    parser.add_argument('--model_save_name', type=str, help='Model save name')
    parser.add_argument('--wandb', type=str, help='Wandb key')
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment', default='UnLiteFlowNet-PIV')
    parser.add_argument('--use_denoising', type=str, help='Denoising method', default='gaussian')
    parser.add_argument('--normalise', type=int, default=1)
    parser.add_argument('--use_pretrained', type=bool, help='Use pretrained model', default=False)
    parser.add_argument('--new_train', type=bool, help='New train', default=True)

    args = parser.parse_args()
    model = test_train(args)
