# 新建文件: evaluate_icm.py

import os
import argparse
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 导入我们的ICM模块
import sys
sys.path.append(".")
sys.path.append("..")
from babybench_selftouch.icm.icm_module import ICMModule

def plot_reconstructions(model, data_loader, device, num_samples=5):
    """可视化原始观测和模型的重构结果"""
    model.eval()
    
    # 从数据加载器中获取一个批次
    p_obs, t_obs, _, _, _ = next(iter(data_loader))
    p_obs, t_obs = p_obs.to(device), t_obs.to(device)

    # 通过模型进行重构
    with torch.no_grad():
        p_recon, _, _, _ = model.proprio_vae(p_obs)
        t_recon, _, _, _ = model.touch_vae(t_obs)

    # 将数据移回CPU并转为NumPy
    p_obs, p_recon = p_obs.cpu().numpy(), p_recon.cpu().numpy()
    t_obs, t_recon = t_obs.cpu().numpy(), t_recon.cpu().numpy()
    
    print("Plotting reconstructions...")
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))
    fig.suptitle('Proprioception VAE: Original vs. Reconstructed', fontsize=16)
    for i in range(num_samples):
        axes[i, 0].plot(p_obs[i])
        axes[i, 0].set_title(f'Sample {i+1} - Original')
        axes[i, 1].plot(p_recon[i])
        axes[i, 1].set_title(f'Sample {i+1} - Reconstructed')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))
    fig.suptitle('Touch VAE: Original vs. Reconstructed', fontsize=16)
    for i in range(num_samples):
        axes[i, 0].bar(range(t_obs.shape[1]), t_obs[i])
        axes[i, 0].set_title(f'Sample {i+1} - Original')
        axes[i, 1].bar(range(t_recon.shape[1]), t_recon[i])
        axes[i, 1].set_title(f'Sample {i+1} - Reconstructed')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def visualize_latent_space(model, data_loader, device):
    """使用t-SNE可视化隐空间"""
    model.eval()
    
    all_z_proprio = []
    all_z_touch = []
    all_has_touch = []

    print("Encoding all test data for t-SNE visualization...")
    with torch.no_grad():
        for p_obs, t_obs, _, _, _ in tqdm(data_loader, desc="Encoding"):
            p_obs, t_obs = p_obs.to(device), t_obs.to(device)
            _, p_mu, _, z_p = model.proprio_vae(p_obs)
            _, t_mu, _, z_t = model.touch_vae(t_obs)
            
            all_z_proprio.append(p_mu.cpu().numpy()) # 使用mu作为稳定的表征
            all_z_touch.append(t_mu.cpu().numpy())
            all_has_touch.append(t_obs.sum(dim=1).cpu().numpy() > 0)

    all_z_proprio = np.concatenate(all_z_proprio, axis=0)
    all_z_touch = np.concatenate(all_z_touch, axis=0)
    all_has_touch = np.concatenate(all_has_touch, axis=0)
    
    print("Running t-SNE... (this may take a while)")
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
    
    # 可视化本体感觉隐空间
    proprio_tsne = tsne.fit_transform(all_z_proprio)
    plt.figure(figsize=(10, 8))
    plt.scatter(proprio_tsne[all_has_touch, 0], proprio_tsne[all_has_touch, 1], s=5, c='r', label='With Touch')
    plt.scatter(proprio_tsne[~all_has_touch, 0], proprio_tsne[~all_has_touch, 1], s=5, c='b', alpha=0.3, label='No Touch')
    plt.title('t-SNE of Proprioception Latent Space')
    plt.legend()
    plt.show()

    # 可视化触觉隐空间
    touch_tsne = tsne.fit_transform(all_z_touch)
    plt.figure(figsize=(10, 8))
    plt.scatter(touch_tsne[all_has_touch, 0], touch_tsne[all_has_touch, 1], s=5, c='r', label='With Touch')
    plt.scatter(touch_tsne[~all_has_touch, 0], touch_tsne[~all_has_touch, 1], s=5, c='b', alpha=0.3, label='No Touch')
    plt.title('t-SNE of Touch Latent Space')
    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Offline ICM Evaluation Script")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the .h5 dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained best_model.pth')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 1. 加载数据并准备测试集
    with h5py.File(args.data_path, 'r') as hf:
        proprio_obs = hf['proprio_obs'][:]
        # ... (此处省略与训练脚本中完全相同的数据加载和划分逻辑)
    
    dataset_size = proprio_obs.shape[0]
    indices = np.random.permutation(dataset_size)
    train_split = int(dataset_size * 0.8)
    val_split = int(dataset_size * 0.9)
    test_indices = indices[val_split:]
    
    p_obs_test = torch.from_numpy(proprio_obs[test_indices]).float()
    t_obs_test = torch.from_numpy(hf['touch_obs'][test_indices]).float()
    act_test = torch.from_numpy(hf['actions'][test_indices]).float()
    next_p_obs_test = torch.from_numpy(hf['next_proprio_obs'][test_indices]).float()
    next_t_obs_test = torch.from_numpy(hf['next_touch_obs'][test_indices]).float()
    
    test_dataset = TensorDataset(p_obs_test, t_obs_test, act_test, next_p_obs_test, next_t_obs_test)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    # 2. 加载模型
    # 从模型路径中推断出参数
    # (这是一个简化的例子，更好的做法是将超参数保存在一个文件中)
    proprio_dim = p_obs_test.shape[1]
    touch_dim = t_obs_test.shape[1]
    action_dim = act_test.shape[1]
    
    # 假设的超参数，需要与训练时一致
    model = ICMModule(proprio_dim, touch_dim, action_dim, proprio_latent_dim=8, touch_latent_dim=8, hidden_dim=512, device=device).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("Model loaded successfully.")
    
    # 3. 运行评估
    plot_reconstructions(model, test_loader, device)
    visualize_latent_space(model, test_loader, device)


if __name__ == '__main__':
    main()