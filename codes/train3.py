import argparse
import os
import sys
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch import optim



# (NEW) Add the path to the 'model' directory
sys.path.append(os.path.join(os.getcwd(), 'C:/Users/Michael/Desktop/Y2 S2/DATA7903-Data Science Capstone Project 2B/dataset and pre-trained model/pre-trained model'))

from model.trajairnet import TrajAirNet
from model.utils import TrajectoryDataset, seq_collate, loss_func
from test2 import test_and_evaluate, plot_loss_curve,  plot_uncertainty_comparison, cluster_and_save_3d_trajectories,plot_and_save_3d_trajectories  # (NEW) Import functions from test2.py


# (NEW) Store loss values for each epoch
train_losses = []
test_ade_losses = []  # (NEW) Store ADE losses for each epoch
test_fde_losses = []  # (NEW) Store FDE losses for each epoch
ood_ade_losses = []  # (NEW) OOD ADE losses
ood_fde_losses = []  # (NEW) OOD FDE losses
in_dist_msp = []
ood_msp = []




def train():
    # Dataset parameters
    parser = argparse.ArgumentParser(description='Train TrajAirNet model')
    parser.add_argument('--dataset_folder', type=str, default='C:/Users/Michael/Desktop/Y2 S2/DATA7903-Data Science Capstone Project 2B/dataset and pre-trained model/pre-trained model/dataset')
    parser.add_argument('--dataset_name', type=str, default='7days1')
    parser.add_argument('--ood_dataset_name', type=str, default='OOD_test')  # (NEW) OOD dataset
    parser.add_argument('--synthetic_dataset_name', type=str, default='synthetic data')  # Folder for synthetic data
    parser.add_argument('--obs', type=int, default=11) #  The number of time steps in the input trajectory (exp, how much historical data you observe)
    parser.add_argument('--preds', type=int, default=120) # how many steps into the future you are predicting
    parser.add_argument('--preds_step', type=int, default=5)  #  how frequently you are making predictions,Changed from 10 to 5 (more frequent)

    # Network parameters
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--tcn_channel_size', type=int, default=256) # the number of channels in each TCN layer
    parser.add_argument('--tcn_layers', type=int, default=2)  
    parser.add_argument('--tcn_kernels', type=int, default=5) # 4 to 5

    parser.add_argument('--num_context_input_c', type=int, default=2)
    parser.add_argument('--num_context_output_c', type=int, default=7)
    parser.add_argument('--cnn_kernels', type=int, default=2)

    parser.add_argument('--gat_heads', type=int, default=20) # attention heads for GAT, changed from 16 to 20 for more graph-based learning
    parser.add_argument('--graph_hidden', type=int, default=256)  
    parser.add_argument('--dropout', type=float, default=0.05)  # won't change to demonstrate possible overfitting issue, also 0.5 is enough
    parser.add_argument('--alpha', type=float, default=0.2) #stronger regulation
    parser.add_argument('--cvae_hidden', type=int, default=128)
    parser.add_argument('--cvae_channel_size', type=int, default=128)
    parser.add_argument('--cvae_layers', type=int, default=2) # depth of CVAE
    parser.add_argument('--mlp_layer', type=int, default=32) # fully connected layers

    parser.add_argument('--lr', type=float, default=0.001)  # won't change for only 5 epochs

    # (NEW) Add arguments for loading a pre-trained model from a checkpoint and debug
    parser.add_argument('--resume_training', type=bool, default=False)
    parser.add_argument('--resume_epoch', type=int, default=2)  # (NEW) Resume from epoch #2#
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'debug'],
                    help='Mode to run the script in: "train" for full training, "debug" for quicker execution')
    parser.add_argument('--total_epochs', type=int, default=5)
    parser.add_argument('--delim', type=str, default=' ')
    parser.add_argument('--evaluate', type=bool, default=False) 
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--model_pth', type=str, default="C:/Users/Michael/Desktop/Y2 S2/DATA7903-Data Science Capstone Project 2B/dataset and pre-trained model/pre-trained model/saved_models/")
    parser.add_argument('--image_save_dir', type=str, default="C:/Users/Michael/Desktop/Y2 S2/DATA7903-Data Science Capstone Project 2B/dataset and pre-trained model/pre-trained model/saved_models/image")  # (NEW) Add image save dir

    args = parser.parse_args()

    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test and train data
    datapath = os.path.join(args.dataset_folder, args.dataset_name, "processed_data")
    ood_datapath = os.path.join(args.dataset_folder, args.ood_dataset_name, "processed_data") 
    synthetic_datapath = os.path.join(args.dataset_folder, args.synthetic_dataset_name, "processed_data")
    
    print("Loading Train Data from ", os.path.join(datapath, "train"))
    dataset_train = TrajectoryDataset(os.path.join(datapath, "train"), obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim)

    print("Loading Test Data from ", os.path.join(datapath, "test"))
    dataset_test = TrajectoryDataset(os.path.join(datapath, "test"), obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim)

    # (NEW) Load OOD test dataset
    print("Loading OOD Test Data from ", os.path.join(ood_datapath, "test"))
    dataset_ood_test = TrajectoryDataset(os.path.join(ood_datapath, "test"), obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim)
    
    # (NEW) Load synthetic dataset
    print("Loading Synthetic Data from ", os.path.join(synthetic_datapath, "train"))
    synthetic_dataset = TrajectoryDataset(os.path.join(synthetic_datapath, "train"), obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim)
    
    # Combine both datasets
    combined_dataset = ConcatDataset([dataset_train, synthetic_dataset])
    
    # (NEW) Set default values for debug mode
    if args.mode == 'debug':
        print("Debug mode enabled: Reducing data size and skipping some steps for faster execution.")
        args.total_epochs = 1
        dataset_train = torch.utils.data.Subset(dataset_train, range(10))  # Only use 10 samples for training
        dataset_test = torch.utils.data.Subset(dataset_test, range(5))  # Use only 5 samples for testing
        dataset_ood_test = torch.utils.data.Subset(dataset_ood_test, range(5))  # Use only 5 samples for OOD testing
        synthetic_dataset = torch.utils.data.Subset(synthetic_dataset, range(5))  # Subset synthetic dataset 
        combined_dataset = ConcatDataset([dataset_train, synthetic_dataset])  # Recombine the dataset 
        


    loader_test = DataLoader(dataset_test, batch_size=16, num_workers=4, shuffle=True, collate_fn=seq_collate)
    loader_ood_test = DataLoader(dataset_ood_test, batch_size=16, num_workers=4, shuffle=True, collate_fn=seq_collate) 
    loader_synthetic = DataLoader(synthetic_dataset, batch_size=16, num_workers=4, shuffle=True, collate_fn=seq_collate)
    # Combine real-world and synthetic data loaders for training
    loader_train = DataLoader(combined_dataset, batch_size=16, num_workers=4, shuffle=True, collate_fn=seq_collate)
    

    model = TrajAirNet(args)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Skip saving the model in debug mode
    if args.mode == 'debug':
        args.save_model = False  # Disable saving the model in debug mode



    # (NEW) Load pre-trained model if resume_training is True
    if args.resume_training:
        model_path = args.model_pth + "model_" + args.dataset_name + "_" + str(args.resume_epoch) + ".pt"
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = args.resume_epoch + 1  # Start from the next epoch
        print(f"Resuming training from epoch {starting_epoch}")
    else:
        starting_epoch = 1  # Start from scratch

    

    

    for epoch in range(starting_epoch, args.total_epochs + 1):
        model.train()
        tot_loss = 0
        tot_batch_count = 0
        print("Starting Training....")
        # Loop over batches in the training data
        for batch in tqdm(loader_train):
            tot_batch_count += 1  # Keep track of total number of batches

            # Move tensors to the device (GPU/CPU)
            batch = [tensor.to(device) for tensor in batch]
            obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start = batch
            num_agents = obs_traj.shape[1]
            pred_traj = torch.transpose(pred_traj, 1, 2)
            adj = torch.ones((num_agents, num_agents)).to(device)  # Make sure adj is on the right device

            optimizer.zero_grad()

            # Forward pass
            recon_y, m, var = model(torch.transpose(obs_traj, 1, 2), pred_traj, adj[0], torch.transpose(context, 1, 2))

             # Calculate loss for each agent and sum the losses
            loss = sum(loss_func(recon_y[agent], torch.transpose(pred_traj[:, :, agent], 0, 1).unsqueeze(0), m[agent], var[agent]) for agent in range(num_agents))

            tot_loss += loss.item()  # Accumulate total loss for epoch
            
            # Perform backward and optimizer step
            loss.backward()
            optimizer.step()
        

            

        avg_train_loss = tot_loss / tot_batch_count
        train_losses.append(avg_train_loss)   
            

        print(f"EPOCH: {epoch} Train Loss: {avg_train_loss}")


        # (NEW) Conditional model saving: Skip saving in debug mode
        if args.mode != 'debug' and args.save_model:
            # Assign model_path inside the condition
            model_path = os.path.join(args.model_pth, f"model_{args.dataset_name}_{epoch}.pt")

            # Ensure the save directory exists
            os.makedirs(args.model_pth, exist_ok=True)  # Ensure the directory exists

            print(f"Saving model at {model_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': tot_loss / tot_batch_count, 
            }, model_path)

        

        # Evaluate on test and OOD data each epoch but don't plot yet
        print(f"Testing model at epoch {epoch}....")
        
        test_ade_loss, test_fde_loss, test_msp, ood_ade_loss, ood_fde_loss, ood_msp_value, real_traj_in_dist, pred_traj_in_dist = test_and_evaluate(
            model, loader_test, loader_ood_test, device, args.image_save_dir, plot=False)


        test_ade_losses.append(test_ade_loss)
        test_fde_losses.append(test_fde_loss)
        in_dist_msp.append(test_msp)
        
        # add the OOD data at each epoch
        ood_ade_losses.append(ood_ade_loss)
        ood_fde_losses.append(ood_fde_loss)
        ood_msp.append(ood_msp_value)
        
        

    # After all epochs are finished, plot the loss and uncertainty curves
    print(f"Final Epoch: Plotting results...")

    # Plot the 3D trajectories for the last epoch
    print("Plotting 3D trajectories...")
    max_plots = 4  # Limit the number of samples for plotting
    for idx in range(min(max_plots, len(real_traj_in_dist))):
        plot_and_save_3d_trajectories(real_traj_in_dist[idx], pred_traj_in_dist[idx], idx, args.image_save_dir)

    # Perform and plot clustering after the final epoch (in-distribution)
    num_clusters = 3  # adjustable
    cluster_and_save_3d_trajectories(pred_traj_in_dist, num_clusters, args.image_save_dir, name='Predicted')  
    cluster_and_save_3d_trajectories(real_traj_in_dist, num_clusters, args.image_save_dir, name='Real')  


    # Plot the loss curves after training
    plot_loss_curve(train_losses, test_ade_losses, test_fde_losses, ood_ade_losses, ood_fde_losses, args.image_save_dir, starting_epoch)

    # Plot uncertainty comparison (separate from loss curve)
    plot_uncertainty_comparison(in_dist_msp, ood_msp,  args.image_save_dir)

    # Print all epoch training losses at the end
    print("Training completed.")
    for i, loss in enumerate(train_losses, start=starting_epoch):
        print(f"Epoch {i}: Train Loss = {loss}")

    # Print Test ADE, FDE, and MSP losses 
    print("Test Results for Each Epoch:")
    for i, (ade_loss, fde_loss, msp) in enumerate(zip(test_ade_losses, test_fde_losses, in_dist_msp), start=starting_epoch):
        print(f"Epoch {i}: Test ADE Loss = {ade_loss}, Test FDE Loss = {fde_loss}, Test Uncertainty (MSP) = {msp}")

    # Print OOD ADE, FDE, and MSP losses 
    print("OOD Test Results for Each Epoch:")
    for i, (ade_loss, fde_loss, msp) in enumerate(zip(ood_ade_losses, ood_fde_losses, ood_msp), start=starting_epoch):
        print(f"Epoch {i}: OOD ADE Loss = {ade_loss}, OOD FDE Loss = {fde_loss}, OOD Uncertainty (MSP) = {msp}")

if __name__ == '__main__':
    train()
