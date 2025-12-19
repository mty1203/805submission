# Reproducing the GNN-VAE-RG Experiment for the 2D Ising Model

This directory contains all code necessary to reproduce the experiments described in the extended abstract.

## Overview

This project trains a hierarchical Graph Neural Network Variational Autoencoder (GNN-VAE) with a Renormalization Group (RG) consistency loss to learn RG-like latent variables from 2D Ising model configurations. The model learns a latent representation that:
- Correlates with physical observables (magnetization, energy density)
- Exhibits scale-invariance under coarse-graining
- Aligns with the relevant eigen-direction of an empirical RG map

## Requirements

### Python Environment

Python 3.8+ is required.

### Core Dependencies

Install the following packages:

```bash
pip install torch>=2.0.0
pip install torch-geometric>=2.3.0
pip install numpy>=1.24.0
pip install matplotlib>=3.7.0
pip install scikit-learn>=1.3.0
```

### PyTorch Geometric Extensions

**Important:** The hierarchical RG encoder uses `torch_geometric.nn.pool.graclus`, which requires `torch-cluster`. This package must be installed separately and must match your PyTorch and CUDA versions.

**Check your PyTorch version:**
```bash
python -c "import torch; print(torch.__version__)"
```

**Install torch-cluster (example for CPU):**
```bash
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

**For CUDA 11.8:**
```bash
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

**For CUDA 12.1:**
```bash
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu121.html
```

Replace `2.0.0` with your actual PyTorch version. See the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for details.

### Additional Dependencies

```bash
pip install torch-scatter torch-sparse torch-spline-conv -f https://data.pyg.org/whl/torch-<YOUR_VERSION>+<CUDA>.html
```

## Directory Structure

```
experiment/
├── README.md                    # This file
├── train_gnn_vae_rg.py         # Main training script
├── analyze_all.py               # Comprehensive analysis script
├── ising_graph_dataset.py       # Dataset class for Ising model
└── requirements.txt             # Dependency list (if provided)
```

## Step-by-Step Reproduction

### Step 1: Generate or Load Data

The dataset is automatically generated on first run if `ising_graph_data.pt` does not exist. The default settings generate:
- Lattice size: 16×16
- Temperature range: K ∈ [0.1, 0.7] (K = J/T, Kc ≈ 0.4407)
- 200 samples per temperature point
- Total: ~5200 configurations

To regenerate data (if needed):
```python
from ising_graph_dataset import IsingGraphDataset
dataset = IsingGraphDataset(
    L=16,
    K_list=[0.1, 0.15, ..., 0.7],  # Your desired K values
    num_samples_per_K=200,
    sweeps_equil=500,
    sweeps_between=10,
    data_path="ising_graph_data.pt",
    force_regenerate=True  # Force regeneration
)
```

### Step 2: Train the Model

Run the training script:

```bash
python train_gnn_vae_rg.py
```

**Default hyperparameters:**
- Lattice size: L = 16
- Hidden dimension: 64
- Latent dimension: 4
- Token dimension: 16
- Number of RG steps: 2 (16→8→4 coarse-graining)
- Batch size: 32
- Learning rate: 1e-3
- Epochs: 60
- β (KL weight): 0.1
- λ_RG (RG loss weight): 0.5

**Output:**
- Trained model saved as `gnn_vae_rg_ising.pt`

**Training progress:**
The script prints loss values for each epoch:
```
Epoch   1  total=0.9343  VAE=0.9283  RG=0.0120  beta_t=0.1000
Epoch   2  total=0.7043  VAE=0.7032  RG=0.0021  beta_t=0.1000
...
```

### Step 3: Analyze Results

After training, run the analysis script to generate all figures:

```bash
python analyze_all.py
```

**Important:** The analysis script requires the model architecture parameters to match those used during training. If you modified hyperparameters, update the corresponding constants in `analyze_all.py`:
- `HIDDEN_DIM`
- `LATENT_DIM`
- `TOKEN_DIM`
- `NUM_RG_STEPS`
- `USE_GNN_DECODER`
- `RECON_LOSS_TYPE`

**Output:**
All analysis plots are saved to `analysis_plots/`:
- `01_pc1_vs_K.png` - PC1 vs coupling K
- `02_pc1_vs_m.png` - PC1 vs magnetization
- `03_pc1_vs_energy.png` - PC1 vs energy density
- `04_latent_pca.png` - Latent space PCA visualization
- `05_latent_diff_vs_K.png` - Scale invariance check
- `06_rg_trajectories.png` - RG flow trajectories
- `07_fixed_point_convergence.png` - Fixed point convergence
- `08_rg_map_analysis.png` - Linear RG map and eigenvalue spectrum
- `09_physical_observables.png` - Physical observables vs K
- `10_rg_flow_diagram.png` - RG flow diagram

## Customizing Experiments

### Modify Training Hyperparameters

Edit the `train_gnn_vae_rg` function call in `train_gnn_vae_rg.py`:

```python
if __name__ == "__main__":
    train_gnn_vae_rg(
        latent_dim=8,              # Increase latent dimension
        beta_anneal_epochs=10,     # Enable KL annealing
        recon_loss_type="bce",     # Use Bernoulli likelihood (better for binary spins)
        use_gnn_decoder=True,      # Use GNN decoder instead of MLP
        # ... other parameters
    )
```

### Modify Model Architecture

The model supports several configuration options:
- `use_hierarchical_rg`: Use hierarchical RG encoder (default: True)
- `num_rg_steps`: Number of coarse-graining steps (default: 2)
- `use_gnn_decoder`: Use GNN decoder vs MLP decoder (default: False)
- `recon_loss_type`: "mse" or "bce" (default: "mse")

### Modify RG Loss

- `use_mu_for_rg_loss`: Use μ instead of z for RG loss (default: True, more stable)
- `use_multilevel_rg_loss`: Use multi-level RG loss (default: True)

## Troubleshooting

### Error: "ImportError: 'graclus' requires 'torch-cluster'"

**Solution:** Install `torch-cluster` matching your PyTorch version (see Requirements section).

### Error: Model state dict mismatch when loading

**Solution:** Ensure all model architecture parameters in `analyze_all.py` match those used during training.

### CUDA out of memory

**Solution:** Reduce `batch_size` or use CPU:
```python
device = "cpu"  # Force CPU usage
```

### Data generation is slow

**Solution:** The dataset uses multi-threaded generation by default. You can adjust `num_workers` in `IsingGraphDataset.__init__`.

## Expected Runtime

- **Data generation:** ~2-5 minutes (single-threaded), ~30 seconds (multi-threaded)
- **Training:** ~5-10 minutes (CPU), ~2-5 minutes (GPU)
- **Analysis:** ~1-2 minutes

## Citation

If you use this code, please cite:

```bibtex
@article{meng2024learning,
  title={Learning Renormalization-Group-Like Latent Variables with a Hierarchical GNN-VAE for the 2D Ising Model},
  author={Meng, Tingyu},
  journal={Physics 805 Extended Abstract},
  year={2024}
}
```

## License

This code is provided for research purposes.

## Contact

For questions or issues, please refer to the main repository.

