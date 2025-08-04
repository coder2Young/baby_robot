import os
import argparse
import torch
import numpy as np
import h5py
from torch.utils.data import TensorDataset, DataLoader, Dataset 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
sys.path.append(".")
sys.path.append("..")
from babybench_selftouch.icm.icm_module import ICMModule


class HDF5Dataset(Dataset):
    """
    An HDF5 dataset that supports lazy loading for multiprocessing.
    The file handle is opened only within the worker process.
    """
    def __init__(self, h5_path, indices):
        self.h5_path = h5_path
        self.indices = indices
        self.h5_file = None
        self.keys = ['proprio_obs', 'touch_obs', 'actions', 'next_proprio_obs', 'next_touch_obs']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        actual_idx = self.indices[index]
        sample = [torch.from_numpy(self.h5_file[key][actual_idx]) for key in self.keys]
        
        return tuple(sample)


def train_icm_offline(args):
    """Main training function for ICM with lazy loading."""
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    log_run_path = os.path.join(args.log_dir, f"run_{args.exp_name}")
    os.makedirs(log_run_path, exist_ok=True)
    writer = SummaryWriter(log_dir=log_run_path)

    print(f"Preparing lazy-loading dataset from {args.data_path}...")
    
    with h5py.File(args.data_path, 'r') as hf:
        dataset_size = hf['actions'].shape[0]
        proprio_dim = hf['proprio_obs'].shape[1]
        touch_dim = hf['touch_obs'].shape[1]
        action_dim = hf['actions'].shape[1]
    
    print(f"Dataset contains {dataset_size} samples.")

    indices = np.random.permutation(dataset_size)
    train_split = int(dataset_size * 0.8)
    val_split = int(dataset_size * 0.9)
    train_indices, val_indices = indices[:train_split], indices[train_split:val_split]

    train_dataset = HDF5Dataset(h5_path=args.data_path, indices=train_indices)
    val_dataset = HDF5Dataset(h5_path=args.data_path, indices=val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = ICMModule(
        proprio_obs_dim=proprio_dim,
        touch_obs_dim=touch_dim,
        action_dim=action_dim,
        proprio_latent_dim=args.proprio_latent_dim,
        touch_latent_dim=args.touch_latent_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        device=device
    ).to(device)
    print("ICM Model initialized.")

    best_val_loss = float('inf')
    
    for epoch in range(args.n_epochs):
        model.train()
        train_losses = {}
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.n_epochs} [Train]")
        
        for p_obs, t_obs, act, next_p_obs, next_t_obs in train_pbar:
            p_obs, t_obs, act = p_obs.to(device), t_obs.to(device), act.to(device)
            next_p_obs, next_t_obs = next_p_obs.to(device), next_t_obs.to(device)
            
            model.optimizer.zero_grad()
            (z_combined, 
             (proprio_recon, p_mu, p_logvar), 
             (touch_recon, t_mu, t_logvar)) = model._encode_and_combine(p_obs, t_obs)
            proprio_vae_total_loss, p_recon_loss, p_kl_loss = model.proprio_vae.compute_loss(p_obs, proprio_recon, p_mu, p_logvar)
            touch_vae_total_loss, t_recon_loss, t_kl_loss = model.touch_vae.compute_loss(t_obs, touch_recon, t_mu, t_logvar)
            z_next_combined = model.encode_states_for_inference(next_p_obs, next_t_obs)
            z_pred = model.forward_model(z_combined, act)
            forward_loss = model.forward_model.compute_loss(z_pred, z_next_combined.detach()) 
            mu_a, log_sigma_a = model.inverse_model(z_combined, z_next_combined.detach())
            inverse_nll_loss = model.inverse_model.compute_loss(mu_a, log_sigma_a, act)
            icm_dynamics_loss = (1 - 0.2) * inverse_nll_loss + 0.2 * forward_loss
            total_loss = (1.0 * proprio_vae_total_loss) + (5.0 * touch_vae_total_loss) + (1.0 * icm_dynamics_loss)
            total_loss.backward()
            model.optimizer.step()
            for k, v in {
                'total_loss': total_loss.item(), 'proprio_recon': p_recon_loss.item(), 'proprio_kl': p_kl_loss.item(),
                'touch_recon': t_recon_loss.item(), 'touch_kl': t_kl_loss.item(), 'forward_loss': forward_loss.item(),
                'inverse_nll': inverse_nll_loss.item()
            }.items():
                train_losses.setdefault(k, []).append(v)
        
        for k, v_list in train_losses.items():
            writer.add_scalar(f'Loss/train_{k}', np.mean(v_list), epoch)

        model.eval()
        val_losses = {}
        with torch.no_grad():
            for p_obs, t_obs, act, next_p_obs, next_t_obs in val_loader:
                p_obs, t_obs, act = p_obs.to(device), t_obs.to(device), act.to(device)
                next_p_obs, next_t_obs = next_p_obs.to(device), next_t_obs.to(device)
                (z_combined, (proprio_recon, p_mu, p_logvar), (touch_recon, t_mu, t_logvar)) = model._encode_and_combine(p_obs, t_obs)
                proprio_vae_total_loss, _, _ = model.proprio_vae.compute_loss(p_obs, proprio_recon, p_mu, p_logvar)
                touch_vae_total_loss, _, _ = model.touch_vae.compute_loss(t_obs, touch_recon, t_mu, t_logvar)
                z_next_combined = model.encode_states_for_inference(next_p_obs, next_t_obs)
                z_pred = model.forward_model(z_combined, act)
                forward_loss = model.forward_model.compute_loss(z_pred, z_next_combined)
                mu_a, log_sigma_a = model.inverse_model(z_combined, z_next_combined)
                inverse_nll_loss = model.inverse_model.compute_loss(mu_a, log_sigma_a, act)
                icm_dynamics_loss = (1 - 0.2) * inverse_nll_loss + 0.2 * forward_loss
                total_loss = (1.0 * proprio_vae_total_loss) + (5.0 * touch_vae_total_loss) + (1.0 * icm_dynamics_loss)
                val_losses.setdefault('total_loss', []).append(total_loss.item())

        avg_val_loss = np.mean(val_losses['total_loss'])
        writer.add_scalar('Loss/Validation_Total', avg_val_loss, epoch)
        print(f"Epoch {epoch+1}: Avg Train Loss: {np.mean(train_losses['total_loss']):.4f}, Avg Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(log_run_path, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")

    train_dataset.close()
    val_dataset.close()
    writer.close()
    print("Offline training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Offline ICM Training Script with Lazy Loading")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the .h5 dataset')
    parser.add_argument('--log_dir', type=str, default='logs/icm_offline', help='Directory for logs and models')
    parser.add_argument('--exp_name', type=str, default='exp1', help='Name for this specific experiment run')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--proprio_latent_dim', type=int, default=8, help='Latent dimension for proprioception VAE')
    parser.add_argument('--touch_latent_dim', type=int, default=8, help='Latent dimension for touch VAE')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension for all models')
    args = parser.parse_args()
    train_icm_offline(args)