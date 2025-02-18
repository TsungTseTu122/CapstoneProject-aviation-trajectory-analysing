import argparse
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt  # (NEW) For plotting
from mpl_toolkits.mplot3d import Axes3D  # (NEW) For 3D plotting
from sklearn.cluster import KMeans  # (NEW) For clustering
import sys  # (NEW) Add personal directory
import traceback  # (NEW) For detailed error tracing
# from sklearn.covariance import EmpiricalCovariance
sys.path.append("C:/Users/Michael/Desktop/Y2 S2/DATA7903-Data Science Capstone Project 2B/dataset and pre-trained model/pre-trained model/")

import torch
from torch.utils.data import DataLoader
from model.trajairnet import TrajAirNet
from model.utils import ade, fde, TrajectoryDataset, seq_collate
import torch.nn.functional as F


# (NEW) Function to plot and save 3D real vs. predicted trajectories
def plot_and_save_3d_trajectories(real_traj, pred_traj, idx, save_dir):
    try:
        print(f"Plotting 3D trajectories for sample {idx}...")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(real_traj[0, :], real_traj[1, :], real_traj[2, :], 'g-', label='Real Trajectory')
        ax.plot(pred_traj[0, :], pred_traj[1, :], pred_traj[2, :], 'r--', label='Predicted Trajectory')
        ax.legend()
        ax.set_title(f'3D Trajectory Comparison (Sample {idx})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        file_path = os.path.join(save_dir, f"3d_trajectory_{idx}.png")
        plt.savefig(file_path)
        print(f"Saved 3D trajectory image: {file_path}")
        plt.close()
    except Exception as e:
        print(f"Error in plot_and_save_3d_trajectories: {e}")
        traceback.print_exc()

# (NEW) Function to perform 3D K-means clustering and save results
def cluster_and_save_3d_trajectories(trajectories, num_clusters, save_dir, name):
    try:
        print(f"Starting 3D clustering for {name} trajectories...")
        trajectories = np.array(trajectories)
        num_samples, num_timesteps, num_features = trajectories.shape
        flattened_trajectories = trajectories.reshape(num_samples, -1)

        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        clusters = kmeans.fit_predict(flattened_trajectories)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for cluster in np.unique(clusters):
            cluster_indices = np.where(clusters == cluster)
            cluster_trajs = flattened_trajectories[cluster_indices]

            for traj in cluster_trajs:
                traj = traj.reshape(num_timesteps, num_features)
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f"Cluster {cluster}")

        ax.set_title(f'3D Trajectory Clustering ({name})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        file_path = os.path.join(save_dir, f"3d_cluster_{name}.png")
        plt.savefig(file_path)
        print(f"Saved 3D cluster image: {file_path}")
        plt.close()

        return clusters
    except Exception as e:
        print(f"Error in cluster_and_save_3d_trajectories: {e}")
        traceback.print_exc()

# (NEW) Function to plot and save loss curves
def plot_loss_curve(train_losses, test_ade_losses, test_fde_losses, ood_ade_losses, ood_fde_losses, save_dir, start_epoch):
    try:
        print("Plotting loss curves...")

        # Dynamically generate epoch numbers starting from the resume epoch or starting epoch
        epochs = np.arange(start_epoch, start_epoch + len(train_losses))  # Start from the desired epoch

        plt.figure()
        plt.plot(epochs, train_losses, label="Train Loss")
        if test_ade_losses:  # Ensure there are values to plot
            plt.plot(epochs, test_ade_losses[:len(epochs)], label="Test ADE Loss")
        if test_fde_losses:
            plt.plot(epochs, test_fde_losses[:len(epochs)], label="Test FDE Loss")
        #if uncertainty_scores:
            #plt.plot(epochs, uncertainty_scores[:len(epochs)], label="Uncertainty (Mahalanobis Distance)")
        
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Train and Test Loss Over Epochs")

        # Ensure that x-axis (epochs) displays only integers
        plt.xticks(epochs)  

        plt.savefig(os.path.join(save_dir, "loss_curve.png"))
        print(f"Saved loss curve plot at {os.path.join(save_dir, 'loss_curve.png')}")
        plt.close()
       
        # Plot comparison of in-distribution test vs OOD test
        print("Plotting comparison of in-distribution vs OOD losses...")

        # Generate the correct epoch numbers dynamically based on the start_epoch
        epochs = np.arange(start_epoch, start_epoch + len(test_ade_losses))  # Start from the desired epoch

        
        # Plot ADE Loss comparison
        plt.figure()
        if test_ade_losses:
            plt.plot(epochs, test_ade_losses[:len(epochs)], label="In-distribution ADE Loss", color='blue')
        if ood_ade_losses:
            plt.plot(epochs, ood_ade_losses[:len(epochs)], label="OOD ADE Loss", linestyle='--', color='red')
        
        plt.xlabel("Epochs")
        plt.ylabel("ADE Loss")
        plt.legend()
        plt.title("In-distribution vs OOD ADE Loss Over Epochs")

        # Set x-axis ticks to integers only
        plt.xticks(np.arange(min(epochs), max(epochs)+1, 1))

        # Save the ADE comparison plot
        plt.savefig(os.path.join(save_dir, "in_dist_vs_ood_ade_loss.png"))
        print(f"Saved in-distribution vs OOD ADE loss plot at {os.path.join(save_dir, 'in_dist_vs_ood_ade_loss.png')}")
        plt.close()

        # Plot FDE Loss comparison
        plt.figure()
        if test_fde_losses:
            plt.plot(epochs, test_fde_losses[:len(epochs)], label="In-distribution FDE Loss", color='blue')
        if ood_fde_losses:
            plt.plot(epochs, ood_fde_losses[:len(epochs)], label="OOD FDE Loss", linestyle='--', color='red')
        
        plt.xlabel("Epochs")
        plt.ylabel("FDE Loss")
        plt.legend()
        plt.title("In-distribution vs OOD FDE Loss Over Epochs")

        # Set x-axis ticks to integers only
        plt.xticks(np.arange(min(epochs), max(epochs)+1, 1))

        # Save the FDE comparison plot
        plt.savefig(os.path.join(save_dir, "in_dist_vs_ood_fde_loss.png"))
        print(f"Saved in-distribution vs OOD FDE loss plot at {os.path.join(save_dir, 'in_dist_vs_ood_fde_loss.png')}")
        plt.close()

    except Exception as e:
        print(f"Error in plot_loss_curve: {e}")
        traceback.print_exc()

# Function to compute Mahalanobis distance
#def mahalanobis_distance(x, mean, cov_inv):
 #   diff = x - mean
  #  return np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))

## Function to plot uncertainty comparison (MSP)
def plot_uncertainty_comparison(in_dist_msp, ood_msp, save_dir):
    try:
        print("Plotting MSP uncertainty comparison (in-distribution vs OOD)...")

        plt.figure(figsize=(8, 6))

        # Plot in-distribution MSP uncertainty
        plt.plot(in_dist_msp, label='In-Distribution MSP', color='blue')
        plt.plot(ood_msp, label='OOD MSP', linestyle='--', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('MSP Uncertainty')
        plt.legend()
        plt.title('MSP Uncertainty Comparison')

        # Save the plot
        uncertainty_path = os.path.join(save_dir, "msp_uncertainty_comparison.png")
        plt.savefig(uncertainty_path)
        print(f"Saved MSP uncertainty comparison plot at {uncertainty_path}")
        plt.close()

    except Exception as e:
        print(f"Error in plot_uncertainty_comparison: {e}")
        traceback.print_exc()




def main():
    try:
        parser = argparse.ArgumentParser(description='Test TrajAirNet model with Uncertainty Estimation')
        parser.add_argument('--dataset_folder', type=str, default='/dataset/')
        parser.add_argument('--dataset_name', type=str, default='7days1')
        parser.add_argument('--ood_dataset_name', type=str, default='OOD_test')  # (NEW) Add OOD dataset for comparison
        parser.add_argument('--epoch', type=int, required=True)
        parser.add_argument('--obs', type=int, default=11)
        parser.add_argument('--preds', type=int, default=120)
        parser.add_argument('--preds_step', type=int, default=1) # 10 to 1
        parser.add_argument('--input_channels', type=int, default=3)
        parser.add_argument('--tcn_channel_size', type=int, default=256) 
        parser.add_argument('--tcn_layers', type=int, default=2)
        parser.add_argument('--tcn_kernels', type=int, default=5) # 4 to 5
        parser.add_argument('--num_context_input_c', type=int, default=2)
        parser.add_argument('--num_context_output_c', type=int, default=7)
        parser.add_argument('--cnn_kernels', type=int, default=2)
        parser.add_argument('--gat_heads', type=int, default=20)  # 16 to 20 
        parser.add_argument('--graph_hidden', type=int, default=256) 
        parser.add_argument('--dropout', type=float, default=0.05)
        parser.add_argument('--alpha', type=float, default=0.2)
        parser.add_argument('--cvae_hidden', type=int, default=128)
        parser.add_argument('--cvae_channel_size', type=int, default=128)
        parser.add_argument('--cvae_layers', type=int, default=2)
        parser.add_argument('--mlp_layer', type=int, default=32)
        parser.add_argument('--delim', type=str, default=' ')
        parser.add_argument('--model_dir', type=str, default="C:/Users/Michael/Desktop/Y2 S2/DATA7903-Data Science Capstone Project 2B/dataset and pre-trained model/pre-trained model/saved_models")
        parser.add_argument('--image_save_dir', type=str, default="C:/Users/Michael/Desktop/Y2 S2/DATA7903-Data Science Capstone Project 2B/dataset and pre-trained model/pre-trained model/saved_models/image/")
        parser.add_argument('--resume_epoch', type=int, required=True, help='Epoch to resume from')

        args = parser.parse_args()

        # Select device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Load in-distribution data (7days1)
        dataset_test, loader_test = load_dataset(args, args.dataset_name, "test", device)

        # Load OOD data (7days2)
        ood_dataset_test, ood_loader_test = load_dataset(args, args.ood_dataset_name, "test", device, is_ood=True)


        # Load model
        model = load_model(args, device)

        # Create the image save directory if it doesn't exist
        if not os.path.exists(args.image_save_dir):
            os.makedirs(args.image_save_dir)
            print(f"Image save directory created at {args.image_save_dir}")

        # Store losses and uncertainty scores for plotting
        train_losses = []  
        test_ade_losses = []
        test_fde_losses = []
    
        test_msp = []  
        
        ood_ade_losses = []
        ood_fde_losses = []

        ood_msp = []  

         # Run test on in-distribution data
        test_ade_loss, test_fde_loss, msp_score = test_and_evaluate(model, loader_test, device, args.image_save_dir)
        print(f"Test ADE Loss: {test_ade_loss}, Test FDE Loss: {test_fde_loss}, MSP: {msp_score}")
        test_ade_losses.append(test_ade_loss)
        test_fde_losses.append(test_fde_loss)
        test_msp.append(msp_score)
    

        # Run test on OOD data
        ood_ade_loss, ood_fde_loss, ood_msp_score = test_and_evaluate(model, ood_loader_test, device, args.image_save_dir)
        print(f"OOD Test ADE Loss: {ood_ade_loss}, OOD Test FDE Loss: {ood_fde_loss}, OOD MSP: {ood_msp_score}")
        ood_ade_losses.append(ood_ade_loss)
        ood_fde_losses.append(ood_fde_loss)
        ood_msp.append(ood_msp_score)
        

        # Plot only on the final epoch by adjusting the condition for `args.epoch`
        if args.epoch == args.total_epochs:  # Plotting happens at the final epoch
            print("Plotting results at the final epoch...")

            # Plot loss curves
            plot_loss_curve(train_losses, test_ade_losses, test_fde_losses, ood_ade_losses, ood_fde_losses, args.image_save_dir, start_epoch=args.resume_epoch)

            # Plot MSP uncertainty comparison
            plot_uncertainty_comparison(test_msp, ood_msp, args.image_save_dir)


    except Exception as e:
        print(f"Error in main function: {e}")
        traceback.print_exc()

def load_dataset(args, dataset_name, split, device, is_ood=False):
    """Load dataset and return DataLoader."""
    try:
        if is_ood:
            # For OOD data (7days2), we don't have separate train/test subdirectories, so use the folder directly
            datapath = os.path.join(args.dataset_folder, dataset_name)
        else:
            # For in-distribution data (7days1), use the usual processed_data/test path
            datapath = os.path.join(args.dataset_folder, dataset_name, "processed_data", split)
        
        print(f"Loading {split if not is_ood else 'OOD'} Data from {datapath}")
        
        dataset = TrajectoryDataset(datapath, obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim)
        loader = DataLoader(dataset, batch_size=16, num_workers=4, shuffle=True, collate_fn=seq_collate) # batch to 16
        
        print(f"{'OOD' if is_ood else split.capitalize()} data loaded successfully for {dataset_name}.")
        return dataset, loader
    except Exception as e:
        print(f"Error loading {'OOD' if is_ood else split} data: {e}")
        traceback.print_exc()


def load_model(args, device):
    """Load model and return the initialized model."""
    try:
        model = TrajAirNet(args)
        model.to(device)
        model_path = os.path.join(os.getcwd(), args.model_dir, f"model_{args.dataset_name}_{str(args.epoch)}.pt")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()





def test_and_evaluate(model, loader_test, loader_ood_test, device, save_dir, plot=False):
    """Evaluate on both in-distribution and OOD data in one function."""
    tot_ade_loss_id = 0
    tot_fde_loss_id = 0
    tot_ade_loss_ood = 0
    tot_fde_loss_ood = 0
    tot_batch_id = 0
    tot_batch_ood = 0
    all_real_traj = []
    all_pred_traj = []
    msp_scores_id = []
    msp_scores_ood = []

    print("Starting evaluation process...")
    # **Evaluate on in-distribution (ID) data**
    for batch in tqdm(loader_test):
        tot_batch_id += 1
        batch = [tensor.to(device) for tensor in batch]
        obs_traj_all, pred_traj_all, obs_traj_rel_all, pred_traj_rel_all, context, seq_start = batch
        num_agents = obs_traj_all.shape[1]
        adj = torch.ones((num_agents, num_agents)).to(device)

        best_ade_loss_id = float('inf')
        best_fde_loss_id = float('inf')

        for i in range(5):  # Take 5 random samples from the latent space
            z = torch.randn([1, 1, 128]).to(device)
            recon_y_all = model.inference(torch.transpose(obs_traj_all, 1, 2), z, adj, torch.transpose(context, 1, 2))

            ade_loss = 0
            fde_loss = 0
            for agent in range(num_agents):
                obs_traj = np.squeeze(obs_traj_all[:, agent, :].cpu().numpy())
                pred_traj = np.squeeze(pred_traj_all[:, agent, :].cpu().numpy())
                recon_pred = np.squeeze(recon_y_all[agent].detach().cpu().numpy()).transpose()

                ade_loss += ade(recon_pred, pred_traj)
                fde_loss += fde(recon_pred, pred_traj)

                all_real_traj.append(pred_traj)
                all_pred_traj.append(recon_pred)

                softmax_probs = F.softmax(torch.tensor(recon_pred), dim=-1)
                max_prob = torch.max(softmax_probs).item()
                msp_scores_id.append(max_prob)

            ade_total_loss_id = ade_loss / num_agents
            fde_total_loss_id = fde_loss / num_agents

            if ade_total_loss_id < best_ade_loss_id:
                best_ade_loss_id = ade_total_loss_id
                best_fde_loss_id = fde_total_loss_id

        tot_ade_loss_id += best_ade_loss_id
        tot_fde_loss_id += best_fde_loss_id

    # **Evaluate on OOD data**
    for batch in tqdm(loader_ood_test):
        tot_batch_ood += 1
        batch = [tensor.to(device) for tensor in batch]
        obs_traj_all, pred_traj_all, obs_traj_rel_all, pred_traj_rel_all, context, seq_start = batch
        num_agents = obs_traj_all.shape[1]
        adj = torch.ones((num_agents, num_agents)).to(device)

        best_ade_loss_ood = float('inf')
        best_fde_loss_ood = float('inf')

        for i in range(5):  # Take 5 random samples from the latent space
            z = torch.randn([1, 1, 128]).to(device)
            recon_y_all = model.inference(torch.transpose(obs_traj_all, 1, 2), z, adj, torch.transpose(context, 1, 2))

            ade_loss = 0
            fde_loss = 0
            for agent in range(num_agents):
                pred_traj = np.squeeze(pred_traj_all[:, agent, :].cpu().numpy())
                recon_pred = np.squeeze(recon_y_all[agent].detach().cpu().numpy()).transpose()

                ade_loss += ade(recon_pred, pred_traj)
                fde_loss += fde(recon_pred, pred_traj)

                softmax_probs = F.softmax(torch.tensor(recon_pred), dim=-1)
                max_prob = torch.max(softmax_probs).item()
                msp_scores_ood.append(max_prob)

            ade_total_loss_ood = ade_loss / num_agents
            fde_total_loss_ood = fde_loss / num_agents

            if ade_total_loss_ood < best_ade_loss_ood:
                best_ade_loss_ood = ade_total_loss_ood
                best_fde_loss_ood = fde_total_loss_ood

        tot_ade_loss_ood += best_ade_loss_ood
        tot_fde_loss_ood += best_fde_loss_ood

    # **Return metrics for both ID and OOD data**
    avg_msp_id = np.mean(msp_scores_id) if msp_scores_id else None
    avg_msp_ood = np.mean(msp_scores_ood) if msp_scores_ood else None

    return (
        tot_ade_loss_id / tot_batch_id, 
        tot_fde_loss_id / tot_batch_id, 
        avg_msp_id, 
        tot_ade_loss_ood / tot_batch_ood, 
        tot_fde_loss_ood / tot_batch_ood, 
        avg_msp_ood, 
        all_real_traj, 
        all_pred_traj
    )







if __name__ == "__main__":
    main()