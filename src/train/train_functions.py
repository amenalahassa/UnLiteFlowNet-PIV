import random
import numpy as np
import json
from src.model.loss_functions import *
from src.model.utils import realEPE
from torch.utils.data import DataLoader
from livelossplot import PlotLosses
import GPUtil
import time
import datetime
from src.model.models import estimate, device
import torch
import wandb
from src.data_processing.helpers import plot_velocity_comparison, compute_rmse, compute_aee, compute_aee_per_x_pixels
from torch.optim.lr_scheduler import StepLR

def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True  # uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms
    torch.backends.cudnn.enabled = True

    return True


def train(model, optimizer, criterion, data_loader):
    """Train function """
    model.train()
    train_loss, flow2_EPE = 0, 0
    iter_num = 0
    total_time = 0
    rmse = 0.
    aee = 0.
    for x, y in data_loader:
        start = time.perf_counter()
        # print(x.shape)
        B, I, H, W = x.shape
        # print(x.shape, y.shape)
        x1 = x[:, 0, ...].view(-1, 1, H, W)
        x2 = x[:, 1, ...].view(-1, 1, H, W)
        y = y.view(-1, 2, H, W)

        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()

        output_forward = estimate(x1, x2, model, train=True)
        output_backward = estimate(x2, x1, model, train=True)
        # print(len(output_forward), len(output_backward), x1.shape, x2.shape)
        loss = criterion(output_forward, output_backward, x1, x2)
        real_timeEPE = realEPE(output_forward[-1], y).item()
        flow2_EPE += real_timeEPE * x.size(0)

        loss.backward()
        train_loss += loss.item() * x.size(0)
        optimizer.step()

        ##-----------------print info-----------------------
        end = time.perf_counter()
        time_used = end - start
        total_time += time_used
        iter_num += 1
        percent = 100 * iter_num * x1.shape[0] / len(data_loader.dataset)
        print(
            "Finished this epoch %1.3f %%, real time EPE loss %1.3f, time used(seconds) %1.3f, expected time to finish %1.3f"
            % (percent, real_timeEPE, total_time,
               (100 - percent) * total_time / percent))
        
        u_pred = output_forward[-1][:, 0, ...].detach().cpu().numpy()
        v_pred = output_forward[-1][:, 1, ...].detach().cpu().numpy()
        u_true = y[:, 0, ...].detach().cpu().numpy()
        v_true = y[:, 1, ...].detach().cpu().numpy()
        rmse += compute_rmse(u_pred, u_true, v_pred, v_true)
        aee += compute_aee(u_pred, u_true, v_pred, v_true)

    return train_loss / len(data_loader.dataset), flow2_EPE / len(
        data_loader.dataset), rmse / len(data_loader.dataset), aee / len(data_loader.dataset), compute_aee_per_x_pixels(aee / len(data_loader.dataset), W)


def validate(model, criterion, data_loader, epoch=0, args=None, set_type='val'):
    """Validate functions"""
    model.eval()
    validation_loss = 0.
    all_u_pred, all_v_pred = [], []
    all_u_true, all_v_true = [], []
    for x, y in data_loader:
        with torch.no_grad():
            B, I, H, W = x.shape
            x1 = x[:, 0, ...].view(-1, 1, H, W)
            x2 = x[:, 1, ...].view(-1, 1, H, W)
            y = y.view(-1, 2, H, W)

            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            output_forward = estimate(x1, x2, model, train=True)
            loss = criterion(output_forward[-1], y)
            validation_loss += loss.item() * x.size(0)
            
                        # Accumulate predictions and ground truths
            all_u_pred.append(output_forward[-1][:, 0, ...].cpu())
            all_v_pred.append(output_forward[-1][:, 1, ...].cpu())
            all_u_true.append(y[:, 0, ...].cpu())
            all_v_true.append(y[:, 1, ...].cpu())

    # Stack all accumulated tensors at the end of the epoch
    all_u_pred = torch.cat(all_u_pred, dim=0).numpy()
    all_v_pred = torch.cat(all_v_pred, dim=0).numpy()
    all_u_true = torch.cat(all_u_true, dim=0).numpy()
    all_v_true = torch.cat(all_v_true, dim=0).numpy()

    if args and args.wandb:
        if epoch % 10 == 0:
            filename = f"./checkpoints/{set_type}_comparison_{epoch}.png"
            plot_velocity_comparison(all_u_true, all_v_true, all_u_pred, all_v_pred, save_path=filename)
            wandb.log({f"{set_type}_comparison": wandb.Image(filename)})
        
    rmse = compute_rmse(all_u_pred, all_u_true, all_v_pred, all_v_true)
    aee = compute_aee(all_u_pred, all_u_true, all_v_pred, all_v_true)

    
    return validation_loss / len(data_loader.dataset), rmse, aee, compute_aee_per_x_pixels(aee, W)


def train_model(model,
                train_dataset,
                validate_dataset,
                test_dataset,
                batch_size,
                test_batch_size,
                lr,
                n_epochs,
                optimizer,
                epoch_trained=0,
                seed=42, args=None):
    """The train function """
    set_seed(seed)

    criterion_train = multiscaleUnsupervisorError
    criterion_validate = realEPE

    # Prepare data loader
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    validation_loader = DataLoader(validate_dataset,
                                   batch_size=test_batch_size,
                                   shuffle=False,
                                   num_workers=4,
                                   pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=test_batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)

    # Add a StepLR scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    liveloss = PlotLosses()
    para_dict = {}
    total_time = 0
    for epoch in range(epoch_trained, n_epochs):
        start_time = time.perf_counter()
        print("Total epoch %d" % n_epochs)
        print("Epoch %d starts! " % epoch)
        print("Memory allocated: ",
              torch.cuda.memory_allocated() / 1024 / 1024 / 1024)

        GPUtil.showUtilization()

        logs = {}
        train_loss, train_loss_epe, train_rmse, train_aee, norm_train_aee = train(model, optimizer, criterion_train,
                                           train_loader)
        validation_loss_epe, validation_rmse, validation_aee, norm_validation_aee = validate(model, criterion_validate,
                                       validation_loader, epoch=epoch+1, args=args)

        end_time = time.perf_counter()
        
        # Step the scheduler
        scheduler.step()

        logs['' + 'multiscale loss'] = train_loss
        logs['' + 'EPE loss'] = train_loss_epe
        logs['val_' + 'EPE loss'] = validation_loss_epe
        liveloss.update(logs)
        liveloss.draw()

        total_time += end_time - start_time

        if args and args.wandb:
            wandb.log({
                "train_loss": train_loss, 
                "train_loss_epe": train_loss_epe,
                "avg_train_rmse": train_rmse,
                "avg_train_aee": train_aee,
                "norm_avg_train_aee": norm_train_aee,
                "validation_rmse": validation_rmse,
                "validation_aee": validation_aee,
                "validation_aee": validation_aee,
                "norm_validation_aee": norm_validation_aee,
                "validation_loss_epe": validation_loss_epe,
                "learning_rate": scheduler.get_last_lr()[0],
            })

        print(
            "Epoch: ", epoch, ", Avg. Train EPE Loss: %1.3f" % train_loss_epe,
            " Avg. Validation EPE Loss: %1.3f" % validation_loss_epe,
            "Time used this epoch (seconds): %1.3f" % (end_time - start_time),
            "Time remain(hrs) %1.3f" % (total_time / (epoch + 1) *
                                        (n_epochs - epoch) / 3600))

        # Every 5 epoach, checkpoint
        if (epoch + 1) % 5 == 0:
            test_loss_epe, test_rmse, test_aee, norm_test_aee = validate(model, criterion_validate, test_loader, epoch=epoch+1, args=args, set_type='test')
            # Fill in the parameters into the dict
            para_dict['epoch'] = epoch
            para_dict['dataset size'] = len(train_loader.dataset)
            para_dict['train EPE'] = train_loss_epe
            para_dict['validation EPE'] = validation_loss_epe
            para_dict['learning rate'] = lr
            para_dict['time used(seconds)'] = total_time

            # There is no actual test loss, so use validation loss here
            para_dict['test EPE'] = test_loss_epe
            

            if args and args.wandb:
                wandb.log({
                    "test_loss_epe": test_loss_epe,
                    "test_rmse": test_rmse,
                    "test_aee": test_aee,
                    "norm_test_aee": norm_test_aee,
                })
                
        if (epoch + 1) % 20 == 0:
            save_model(model, optimizer, train_loss, para_dict,
                       "UnLiteFlowNet_checkpoint_%d_" % epoch, args)

    test_loss_epe, test_rmse, test_aee, norm_test_aee = validate(model, criterion_validate, test_loader, epoch=epoch+1, args=args, set_type='test')
    print(" Avg. Test EPE Loss: %1.3f" % test_loss_epe,
          "Total time used(seconds): %1.3f" % total_time)

    if args and args.wandb:
        wandb.log({
            "test_loss_epe": test_loss_epe,
            "test_rmse": test_rmse,
            "test_aee": test_aee,
            "norm_test_aee": norm_test_aee,
        })
    print("")

    # Fill in the parameters into the dict
    para_dict = {}
    para_dict['epoch'] = n_epochs
    para_dict['dataset size'] = len(train_loader.dataset)
    para_dict['batch_size'] = batch_size
    para_dict['train EPE'] = train_loss_epe
    para_dict['validation EPE'] = validation_loss_epe
    para_dict['learning rate'] = lr
    para_dict['time used(seconds)'] = total_time
    para_dict['test EPE'] = test_loss_epe
    save_model(model, optimizer, train_loss, para_dict,
               "UnLiteFlowNet_%d_" % epoch)

    if args and args.wandb:
        wandb.finish()

    return model


def save_model(model, optimizer, train_loss, para_dict, save_name, args=None):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
    model_save_name = save_name + st + '.pt'
    PATH = F"./checkpoints/{model_save_name}"
    epoch = para_dict['epoch']
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, PATH)

    # Serialize data into file:
    json.dump(para_dict, open("./checkpoints/" + save_name + st + '.json', 'w'))

    if args and args.wandb:
        # wandb.log_model(PATH, save_name)
        artifact = wandb.Artifact(name = "models", type = "model")
        artifact.add_file(local_path = PATH, name = model_save_name)
        artifact.save()
        # wandb.save(save_name + st + '.json')

    return model_save_name
