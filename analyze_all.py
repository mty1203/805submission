# analyze_all.py


import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from train_gnn_vae_rg import (
    NnInteractionTokenizer,
    GNN_VAE_RG,
    block_spin_2x2,
)
from ising_graph_dataset import IsingGraphDataset

DEVICE = "cpu"
MODEL_PATH = "gnn_vae_rg_ising.pt"
DATA_PATH = "ising_graph_data.pt"
OUTPUT_DIR = "analysis_plots"

L = 16
K_LIST =vtgtv
NUM_SAMPLES_PER_K = 200
HIDDEN_DIM = 64
LATENT_DIM = 4
TOKEN_DIM = 16
NUM_RG_STEPS = 2  

Kc = 0.4407 


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_model():
 
    model = GNN_VAE_RG(
        in_dim=TOKEN_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_nodes=L * L,
        use_hierarchical_rg=True,
        num_rg_steps=NUM_RG_STEPS,
    ).to(DEVICE)
    
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


def load_data():

    dataset = IsingGraphDataset(
        L=L,
        K_list=K_LIST,
        num_samples_per_K=NUM_SAMPLES_PER_K,
        sweeps_equil=500,
        sweeps_between=10,
        device=DEVICE,
        periodic=True,
        data_path=DATA_PATH,
        force_regenerate=False,
    )
    return dataset


def compute_magnetization(spins):

    return torch.abs(spins.mean()).item()


def compute_energy_density(spins, L):
    """ e = -J * sum(s_i * s_j) / N"""
    spins_2d = spins.view(L, L)
    energy = 0.0
    for i in range(L):
        for j in range(L):
            # 最近邻相互作用
            energy -= spins_2d[i, j] * spins_2d[(i+1) % L, j]
            energy -= spins_2d[i, j] * spins_2d[i, (j+1) % L]
    return energy / (L * L)


def extract_all_data(model, dataset, tokenizer, max_levels=3):

    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 存储
    mus_per_level = [[] for _ in range(max_levels)]
    K_all = []
    phase_all = []
    m_all = []
    e_all = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            graphs = batch.to_data_list()
            

            for g in graphs:
                spins = g.x.view(-1)
                m_all.append(compute_magnetization(spins))
                e_all.append(compute_energy_density(spins, L))
            
     
            current_graphs = [g.cpu() for g in graphs]
            for level in range(max_levels):
                L_current = int(math.sqrt(current_graphs[0].x.size(0)))
                if L_current < 2:
                    break
                
                batch_level = Batch.from_data_list(current_graphs).to(DEVICE)
                batch_tok = tokenizer(batch_level)
                
     
                try:
                    _, mu, _ = model.encode(batch_tok, return_all_levels=False)
                except:
                    _, mu, _ = model.encode(batch_tok)
                
                mus_per_level[level].append(mu.cpu())
                
                if level < max_levels - 1 and L_current >= 4:
                    current_graphs = [block_spin_2x2(g) for g in current_graphs]
            
            K_all.append(batch.K.view(-1).cpu())
            phase_all.append(batch.y.view(-1).cpu())
    

    for level in range(max_levels):
        if mus_per_level[level]:
            mus_per_level[level] = torch.cat(mus_per_level[level], dim=0).numpy()
        else:
            mus_per_level[level] = None
    
    K_all = torch.cat(K_all, dim=0).numpy()
    phase_all = torch.cat(phase_all, dim=0).numpy()
    m_all = np.array(m_all)
    e_all = np.array(e_all)
    
    return mus_per_level, K_all, phase_all, m_all, e_all


def plot_pc1_vs_K(pca, mu, K_all, phase_all, output_dir):

    pc = pca.transform(mu)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(K_all, pc[:, 0], c=phase_all, cmap='coolwarm', 
                        s=20, alpha=0.6, edgecolors='none')
    ax.axvline(x=Kc, color='green', linestyle='--', linewidth=2, label=f'Kc={Kc}')
    ax.set_xlabel('K = J/T', fontsize=12)
    ax.set_ylabel('PC1', fontsize=12)
    ax.set_title('PC1 vs K (Relevant Direction)', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Phase (0=para, 1=ferro)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_pc1_vs_K.png'), dpi=200)
    print(f"Saved 01_pc1_vs_K.png")
    plt.close()


def plot_pc1_vs_m(pca, mu, m_all, output_dir):

    pc = pca.transform(mu)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(m_all, pc[:, 0], c=m_all, cmap='viridis', 
                        s=20, alpha=0.6, edgecolors='none')
    ax.set_xlabel('|m| (Magnetization)', fontsize=12)
    ax.set_ylabel('PC1', fontsize=12)
    ax.set_title('PC1 vs Magnetization', fontsize=14)
    ax.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('|m|')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_pc1_vs_m.png'), dpi=200)
    print(f"Saved 02_pc1_vs_m.png")
    plt.close()


def plot_pc1_vs_e(pca, mu, e_all, output_dir):

    pc = pca.transform(mu)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(e_all, pc[:, 0], c=e_all, cmap='plasma', 
                        s=20, alpha=0.6, edgecolors='none')
    ax.set_xlabel('Energy Density e', fontsize=12)
    ax.set_ylabel('PC1', fontsize=12)
    ax.set_title('PC1 vs Energy Density', fontsize=14)
    ax.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('e')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_pc1_vs_energy.png'), dpi=200)
    print(f"Saved 03_pc1_vs_energy.png")
    plt.close()


def plot_latent_pca(pca, mu, K_all, phase_all, output_dir):
 
    pc = pca.transform(mu)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
  
    ax = axes[0]
    scatter = ax.scatter(pc[:, 0], pc[:, 1], c=phase_all, cmap='coolwarm', 
                        s=20, alpha=0.6, edgecolors='none')
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('Latent Space (color=phase)', fontsize=14)
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Phase')
    
    
    ax = axes[1]
    scatter = ax.scatter(pc[:, 0], pc[:, 1], c=K_all, cmap='viridis', 
                        s=20, alpha=0.6, edgecolors='none')
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('Latent Space (color=K)', fontsize=14)
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('K = J/T')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_latent_pca.png'), dpi=200)
    print(f"Saved 04_latent_pca.png")
    plt.close()


def plot_latent_diff_vs_K(mus_per_level, K_all, output_dir):
    """5: ||z^(i) - z^(i+1)|| vs K"""
    n_levels = sum(1 for m in mus_per_level if m is not None)
    
    fig, axes = plt.subplots(1, n_levels - 1, figsize=(6 * (n_levels - 1), 5))
    if n_levels - 1 == 1:
        axes = [axes]
    
    for i in range(n_levels - 1):
        if mus_per_level[i] is None or mus_per_level[i + 1] is None:
            continue
        
        diff = mus_per_level[i] - mus_per_level[i + 1]
        norm = np.linalg.norm(diff, axis=1)
        
        ax = axes[i]
        scatter = ax.scatter(K_all, norm, c=K_all, cmap='viridis', 
                            s=15, alpha=0.6, edgecolors='none')
        ax.axvline(x=Kc, color='red', linestyle='--', linewidth=2, label=f'Kc={Kc}')
        ax.set_xlabel('K = J/T', fontsize=12)
        ax.set_ylabel(f'||z^({i}) - z^({i+1})||', fontsize=12)
        ax.set_title(f'Level {i} -> {i+1} Latent Diff', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_latent_diff_vs_K.png'), dpi=200)
    print(f"Saved 05_latent_diff_vs_K.png")
    plt.close()


def plot_rg_trajectories(mus_per_level, K_all, phase_all, output_dir, samples_per_K=5):
    """6: RG Flow Trajectories"""
    # 用 level 0 的 PCA
    pca = PCA(n_components=2)
    pcs = []
    for mu in mus_per_level:
        if mu is not None:
            if len(pcs) == 0:
                pca.fit(mu)
            pcs.append(pca.transform(mu))
        else:
            pcs.append(None)
    
    unique_K = np.unique(K_all)
    

    selected_indices = []
    for K in unique_K:
        K_mask = K_all == K
        K_indices = np.where(K_mask)[0]
        np.random.seed(42)
        chosen = np.random.choice(K_indices, min(samples_per_K, len(K_indices)), replace=False)
        selected_indices.extend(chosen)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    

    ax = axes[0]
    for idx in selected_indices:
        pts = []
        for pc in pcs:
            if pc is not None and idx < pc.shape[0]:
                pts.append(pc[idx])
        if len(pts) < 2:
            continue
        pts = np.array(pts)
        color = 'red' if phase_all[idx] == 1 else 'blue'
        ax.plot(pts[:, 0], pts[:, 1], '-', color=color, alpha=0.3, linewidth=1)
        for i in range(len(pts) - 1):
            ax.annotate('', xy=(pts[i+1, 0], pts[i+1, 1]), 
                       xytext=(pts[i, 0], pts[i, 1]),
                       arrowprops=dict(arrowstyle='->', color=color, alpha=0.5, lw=1))
    

    for pc in pcs:
        if pc is None:
            continue
        para_mask = phase_all == 0
        ferro_mask = phase_all == 1
        if para_mask.sum() > 0:
            para_center = pc[para_mask].mean(axis=0)
            ax.scatter(*para_center, s=200, c='blue', marker='*', edgecolors='black', zorder=10)
        if ferro_mask.sum() > 0:
            ferro_center = pc[ferro_mask].mean(axis=0)
            ax.scatter(*ferro_center, s=200, c='red', marker='*', edgecolors='black', zorder=10)
    
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('RG Trajectories (blue=para, red=ferro, *=fixed points)', fontsize=12)
    ax.grid(alpha=0.3)

    ax = axes[1]
    norm = plt.Normalize(K_all.min(), K_all.max())
    cmap = plt.cm.viridis
    
    for idx in selected_indices:
        pts = []
        for pc in pcs:
            if pc is not None and idx < pc.shape[0]:
                pts.append(pc[idx])
        if len(pts) < 2:
            continue
        pts = np.array(pts)
        color = cmap(norm(K_all[idx]))
        for i in range(len(pts) - 1):
            ax.annotate('', xy=(pts[i+1, 0], pts[i+1, 1]), 
                       xytext=(pts[i, 0], pts[i, 1]),
                       arrowprops=dict(arrowstyle='->', color=color, alpha=0.7, lw=1.5))
    
    sc = ax.scatter(pcs[0][:, 0], pcs[0][:, 1], c=K_all, cmap='viridis', 
                   s=20, alpha=0.5, edgecolors='none')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('K = J/T')
    
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('RG Trajectories (color=K)', fontsize=12)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_rg_trajectories.png'), dpi=200)
    print(f"Saved 06_rg_trajectories.png")
    plt.close()


def plot_fixed_point_convergence(mus_per_level, K_all, phase_all, output_dir):
    """7: Fixed Point Convergence under Coarse-Graining"""
    n_levels = sum(1 for m in mus_per_level if m is not None)
    if n_levels < 1:
        print("Skipping fixed point convergence: no levels")
        return
    

    pca = PCA(n_components=2)
    pca.fit(mus_per_level[0])
    
    fig, axes = plt.subplots(1, n_levels, figsize=(5 * n_levels, 5))
    if n_levels == 1:
        axes = [axes]
    
    L_sizes = [16, 8, 4, 2]  
    
    for lvl in range(n_levels):
        if mus_per_level[lvl] is None:
            continue
        
        ax = axes[lvl]
        pc = pca.transform(mus_per_level[lvl])
        

        para_mask = phase_all == 0
        ferro_mask = phase_all == 1
        
        ax.scatter(pc[para_mask, 0], pc[para_mask, 1], c='blue', s=30, alpha=0.6, label='para (K<Kc)')
        ax.scatter(pc[ferro_mask, 0], pc[ferro_mask, 1], c='red', s=30, alpha=0.6, label='ferro (K>Kc)')
        
     
        if para_mask.sum() > 0:
            para_center = pc[para_mask].mean(axis=0)
            ax.scatter(*para_center, s=300, c='blue', marker='*', edgecolors='black', linewidths=1.5, zorder=10)
        if ferro_mask.sum() > 0:
            ferro_center = pc[ferro_mask].mean(axis=0)
            ax.scatter(*ferro_center, s=300, c='red', marker='*', edgecolors='black', linewidths=1.5, zorder=10)
        
        L_current = L_sizes[lvl] if lvl < len(L_sizes) else '?'
        ax.set_xlabel('PC1', fontsize=11)
        ax.set_ylabel('PC2', fontsize=11)
        ax.set_title(f'Level {lvl} (L={L_current})', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Fixed Point Convergence under Coarse-Graining', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_fixed_point_convergence.png'), dpi=200)
    print(f"Saved 07_fixed_point_convergence.png")
    plt.close()


def plot_rg_map_analysis(mus_per_level, K_all, output_dir):

    if mus_per_level[0] is None or mus_per_level[1] is None:
        print("Skipping RG map analysis: insufficient levels")
        return
    
    mu_fine = mus_per_level[0]
    mu_coarse = mus_per_level[1]
    
    # PCA
    pca = PCA(n_components=2)
    pca.fit(mu_fine)
    pc_fine = pca.transform(mu_fine)
    pc_coarse = pca.transform(mu_coarse)
    

    reg = LinearRegression()
    reg.fit(mu_fine, mu_coarse)
    W = reg.coef_
    b = reg.intercept_
    mu_pred = reg.predict(mu_fine)
    
  
    Kc_local = 0.4407
    mask_away_from_critical = np.abs(K_all - Kc_local) > 0.1
    if mask_away_from_critical.sum() > 50:  
        reg_clean = LinearRegression()
        reg_clean.fit(mu_fine[mask_away_from_critical], mu_coarse[mask_away_from_critical])
        W_clean = reg_clean.coef_
        eigenvalues_clean = np.linalg.eigvals(W_clean)
        print(f"\n[Away from critical] Eigenvalues: {sorted(np.abs(eigenvalues_clean), reverse=True)}")

    eigenvalues = np.linalg.eigvals(W)
    eigen_abs = np.abs(eigenvalues)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 图 7a: Fine vs Coarse
    ax = axes[0]
    ax.scatter(pc_fine[:, 0], pc_fine[:, 1], c=K_all, cmap='viridis', 
              s=30, alpha=0.6, marker='o', label='Fine mu')
    ax.scatter(pc_coarse[:, 0], pc_coarse[:, 1], c=K_all, cmap='viridis', 
              s=30, alpha=0.6, marker='x', label='Coarse mu')
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('Fine vs Coarse Latent', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
 
    ax = axes[1]
    pc_pred = pca.transform(mu_pred)
    ax.scatter(pc_coarse[:, 0], pc_pred[:, 0], c=K_all, cmap='viridis', 
              s=30, alpha=0.6)
    lims = [min(pc_coarse[:, 0].min(), pc_pred[:, 0].min()),
            max(pc_coarse[:, 0].max(), pc_pred[:, 0].max())]
    ax.plot(lims, lims, '--', color='gray', label='y=x')
    ax.set_xlabel('True mu_coarse (PC1)', fontsize=12)
    ax.set_ylabel('Predicted mu\' = F(mu_fine) (PC1)', fontsize=12)
    ax.set_title('RG Map Prediction Accuracy', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    

    ax = axes[2]
    eigen_sorted = sorted(eigen_abs, reverse=True)

    colors = ['red' if i == 0 else 'blue' for i in range(len(eigen_sorted))]
    ax.bar(range(len(eigen_sorted)), eigen_sorted, color=colors)
    ax.axhline(y=1, color='gray', linestyle='--', label='|lambda|=1')
    ax.set_xlabel('Eigenvalue index', fontsize=12)
    ax.set_ylabel('|lambda|', fontsize=12)
    ax.set_title('Eigenvalue Spectrum of RG Map\n(red=dominant, blue=others)', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08_rg_map_analysis.png'), dpi=200)
    print(f"Saved 08_rg_map_analysis.png")
    plt.close()
    

    print(f"\nRG Map Eigenvalues: {sorted(eigen_abs, reverse=True)}")
    

    eigenvalues_full, eigenvectors = np.linalg.eig(W)
    dominant_idx = np.argmax(np.abs(eigenvalues_full))
    dominant_eigenvec = eigenvectors[:, dominant_idx].real  
    pc1_direction = pca.components_[0]  
    
    # Cosine similarity
    cosine_sim = np.abs(np.dot(dominant_eigenvec, pc1_direction)) / (
        np.linalg.norm(dominant_eigenvec) * np.linalg.norm(pc1_direction)
    )
    print(f"Cosine similarity (PC1 vs W's dominant eigenvector): {cosine_sim:.4f}")
    print(f"  - PC1 direction: {pc1_direction}")
    print(f"  - W's dominant eigenvector: {dominant_eigenvec}")


def plot_physical_observables(K_all, m_all, e_all, output_dir):
  
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(K_all, m_all, c=K_all, cmap='viridis', s=15, alpha=0.5)
    ax.axvline(x=Kc, color='red', linestyle='--', linewidth=2, label=f'Kc={Kc}')
    ax.set_xlabel('K = J/T', fontsize=12)
    ax.set_ylabel('|m|', fontsize=12)
    ax.set_title('Magnetization vs K', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
 
    ax = axes[1]
    ax.scatter(K_all, e_all, c=K_all, cmap='viridis', s=15, alpha=0.5)
    ax.axvline(x=Kc, color='red', linestyle='--', linewidth=2, label=f'Kc={Kc}')
    ax.set_xlabel('K = J/T', fontsize=12)
    ax.set_ylabel('Energy Density e', fontsize=12)
    ax.set_title('Energy Density vs K', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '09_physical_observables.png'), dpi=200)
    print(f"Saved 09_physical_observables.png")
    plt.close()


def plot_rg_flow_diagram(mus_per_level, K_all, phase_all, output_dir):
    """"""
    

    pca = PCA(n_components=2)
    pca.fit(mus_per_level[0])
    
    n_levels = sum(1 for m in mus_per_level if m is not None)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
 
    unique_K = np.unique(K_all)

    np.random.seed(42)
    for K in unique_K:
        K_mask = K_all == K
        K_indices = np.where(K_mask)[0]
      
        chosen = np.random.choice(K_indices, min(3, len(K_indices)), replace=False)
        
        for idx in chosen:
            pcs = []
            for lvl in range(n_levels):
                if mus_per_level[lvl] is not None and idx < mus_per_level[lvl].shape[0]:
                    pc = pca.transform(mus_per_level[lvl][idx:idx+1])
                    pcs.append(pc[0, 0]) 
            
            if len(pcs) >= 2:
                color = 'red' if phase_all[idx] == 1 else 'blue'
            
                for i in range(len(pcs) - 1):
                    ax.annotate('', 
                               xy=(pcs[i+1], K),
                               xytext=(pcs[i], K),
                               arrowprops=dict(arrowstyle='->', color=color, alpha=0.3, lw=1))
    
  
    level_colors = ['purple', 'green', 'orange']
    level_labels = ['Level 0 (16x16)', 'Level 1 (8x8)', 'Level 2 (4x4)']
    
    for lvl in range(n_levels):
        if mus_per_level[lvl] is None:
            continue
        
        pc = pca.transform(mus_per_level[lvl])[:, 0]
        
  
        mean_pc1 = []
        for K in unique_K:
            K_mask = K_all == K
            mean_pc1.append(pc[K_mask].mean())
        
        ax.scatter(mean_pc1, unique_K, s=80, c=level_colors[lvl], 
                  marker='o', label=level_labels[lvl], alpha=0.8, edgecolors='black')
        ax.plot(mean_pc1, unique_K, '--', color=level_colors[lvl], alpha=0.5, linewidth=2)
    
  
    ax.axhline(y=Kc, color='green', linestyle=':', linewidth=2, label=f'Kc = {Kc}')
    
   
    ax.axhspan(0, Kc, alpha=0.1, color='blue', label='Paramagnetic phase')
    ax.axhspan(Kc, K_all.max() + 0.05, alpha=0.1, color='red', label='Ferromagnetic phase')
    
   
    ax.text(ax.get_xlim()[0] + 0.1, 0.15, 'Para Fixed Point\n(disordered)', 
           fontsize=10, color='blue', fontweight='bold')
    ax.text(ax.get_xlim()[1] - 0.3, 0.6, 'Ferro Fixed Point\n(ordered)', 
           fontsize=10, color='red', fontweight='bold', ha='right')
    
    ax.set_xlabel('PC1 (Relevant Direction)', fontsize=12)
    ax.set_ylabel('K = J/T (Coupling)', fontsize=12)
    ax.set_title('RG Flow Diagram\nArrows: coarse-graining direction', fontsize=14)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '10_rg_flow_diagram.png'), dpi=200)
    print(f"Saved 10_rg_flow_diagram.png")
    plt.close()


def main():
    ensure_dir(OUTPUT_DIR)
    
    print("=" * 50)

    print("=" * 50)
    


    model = load_model()
    dataset = load_data()
    tokenizer = NnInteractionTokenizer(token_dim=TOKEN_DIM).to(DEVICE)
    


    mus_per_level, K_all, phase_all, m_all, e_all = extract_all_data(
        model, dataset, tokenizer, max_levels=3
    )
    
    n_levels = sum(1 for m in mus_per_level if m is not None)

    

    pca = PCA(n_components=2)
    pca.fit(mus_per_level[0])
    

    
    plot_pc1_vs_K(pca, mus_per_level[0], K_all, phase_all, OUTPUT_DIR)
    plot_pc1_vs_m(pca, mus_per_level[0], m_all, OUTPUT_DIR)
    plot_pc1_vs_e(pca, mus_per_level[0], e_all, OUTPUT_DIR)
    plot_latent_pca(pca, mus_per_level[0], K_all, phase_all, OUTPUT_DIR)
    plot_latent_diff_vs_K(mus_per_level, K_all, OUTPUT_DIR)
    plot_rg_trajectories(mus_per_level, K_all, phase_all, OUTPUT_DIR, samples_per_K=8)
    plot_fixed_point_convergence(mus_per_level, K_all, phase_all, OUTPUT_DIR)
    plot_rg_map_analysis(mus_per_level, K_all, OUTPUT_DIR)
    plot_physical_observables(K_all, m_all, e_all, OUTPUT_DIR)
    plot_rg_flow_diagram(mus_per_level, K_all, phase_all, OUTPUT_DIR)
    
    print("\n" + "=" * 50)

    print("=" * 50)

    print("  01_pc1_vs_K.png              - PC1 vs K (Relevant Direction)")
    print("  02_pc1_vs_m.png              - PC1 vs Magnetization")
    print("  03_pc1_vs_energy.png         - PC1 vs Energy Density")
    print("  04_latent_pca.png            - Latent Space PCA")
    print("  05_latent_diff_vs_K.png      - ||z^(i) - z^(i+1)|| vs K")
    print("  06_rg_trajectories.png       - RG Flow Trajectories")
    print("  07_fixed_point_convergence.png - Fixed Point Convergence")
    print("  08_rg_map_analysis.png       - RG Map Analysis")
    print("  09_physical_observables.png  - Physical Observables vs K")
    print("  10_rg_flow_diagram.png       - RG Flow Diagram (K vs PC1)")


if __name__ == "__main__":
    main()

