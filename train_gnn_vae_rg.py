# train_gnn_vae_rg.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, graclus, max_pool
from torch_geometric.data import Batch, Data

from ising_graph_dataset import IsingGraphDataset, build_square_lattice_edge_index

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- 1. Nearest-Neighbor Interaction Tokenizer ----------


class NnInteractionTokenizer(nn.Module):
    """
    Nearest-neighbor interaction tokenizer:
    Input: Data(x: [N,1], edge_index)
    Output: Data(x: [N, token_dim]), each node's token contains:
      - s_i
      - local_field_i = sum_j s_i * s_j  (local energy / effective field)
    Then pass through a small MLP to map to token_dim dimensions
    """

    def __init__(self, token_dim: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
        )

    def forward(self, data: Data) -> Data:
        # Original spins [N,1]
        x = data.x  # s_i = ±1
        # Ensure tokenizer parameters and input are on the same device (compatible with CPU/GPU)
        device = x.device
        self.mlp = self.mlp.to(device)
        edge_index = data.edge_index
        row, col = edge_index  # row -> source node i, col -> target node j

        # bond_ij = s_i * s_j (nearest-neighbor interaction sign)
        bond = x[row] * x[col]  # [E,1]

        # For each i, accumulate bond_ij over all neighbors j -> local_field_i
        local_field = torch.zeros_like(x)  # [N,1]
        local_field.index_add_(0, row, bond)

        # Concatenate to [s_i, local_field_i]
        feats = torch.cat([x, local_field], dim=1)  # [N,2]

        # Pass through a small MLP to get node token
        tokens = self.mlp(feats)  # [N, token_dim]

        # Replace original x
        data.x = tokens
        return data


# ---------- 2. Hierarchical RG Encoder ----------

class RGBlock(nn.Module):
    """
    One RG step:
      - GCN message passing (local information propagation)
      - graph coarsening via graclus + max_pool (coarse-graining)
    
    16×16 → GCN → pool → 8×8  This step is the RG transformation
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, batch):
        """
        x: [N, in_dim]
        edge_index: [2, E]
        batch: [N] batch index
        Returns: (x_coarse, edge_index_coarse, batch_coarse)
        """
        # 1. GCN message passing (local interactions)
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))

        # 2. Coarse-graining: graclus clustering + max_pool
        cluster = graclus(edge_index, num_nodes=h.size(0))
        data = Data(x=h, edge_index=edge_index, batch=batch)
        data_coarse = max_pool(cluster, data)
        
        return data_coarse.x, data_coarse.edge_index, data_coarse.batch


class HierarchicalRGEncoder(nn.Module):
    """
    Hierarchical RG Encoder (each level outputs μ, σ):
      16×16 → μ⁽⁰⁾,σ⁽⁰⁾ → RGBlock → 8×8 → μ⁽¹⁾,σ⁽¹⁾ → RGBlock → 4×4 → μ⁽²⁾,σ⁽²⁾
    
    Each RGBlock step = GCN convolution + pooling dimension reduction, which is one RG transformation
    Each level can compute latent, used for multi-scale RG consistency loss
    """
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int, num_rg_steps: int = 2):
        super().__init__()
        self.num_rg_steps = num_rg_steps
        self.hidden_dim = hidden_dim
        
        # Initial layer: map input features to hidden_dim (for computing z⁽⁰⁾)
        self.init_conv1 = GCNConv(in_dim, hidden_dim)
        self.init_conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Multiple RGBlocks
        blocks = []
        for i in range(num_rg_steps):
            blocks.append(RGBlock(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList(blocks)
        
        # Each level has its own μ/σ projection layers (total num_rg_steps + 1 levels: 0, 1, ..., num_rg_steps)
        self.mu_layers = nn.ModuleList([
            nn.Linear(hidden_dim, latent_dim) for _ in range(num_rg_steps + 1)
        ])
        self.logvar_layers = nn.ModuleList([
            nn.Linear(hidden_dim, latent_dim) for _ in range(num_rg_steps + 1)
        ])

    def forward(self, data, return_all_levels: bool = False):
        """
        Returns:
          - If return_all_levels=False: (mu_final, logvar_final)
          - If return_all_levels=True: (mus_list, logvars_list)
            where mus_list[i] is the μ of level i
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        mus = []
        logvars = []
        
        # Level 0 (16×16): First do GCN to get hidden representation, then global pool to get μ⁽⁰⁾
        h = F.relu(self.init_conv1(x, edge_index))
        h = F.relu(self.init_conv2(h, edge_index))
        hg = global_mean_pool(h, batch)
        mus.append(self.mu_layers[0](hg))
        logvars.append(self.logvar_layers[0](hg))
        
        # Subsequent RG layers
        x = h  # Use GCN-processed features as input for next layer
        for i, block in enumerate(self.blocks):
            x, edge_index, batch = block(x, edge_index, batch)
            hg = global_mean_pool(x, batch)
            mus.append(self.mu_layers[i + 1](hg))
            logvars.append(self.logvar_layers[i + 1](hg))
        
        if return_all_levels:
            return mus, logvars
        else:
            # Only return final level
            return mus[-1], logvars[-1]


# Keep original simple encoder as alternative
class GNNEncoder(nn.Module):
    """
    Simple GNN Encoder (without hierarchical RG): GNN + global pooling -> mu, logvar
    """
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin_mu = nn.Linear(hidden_dim, latent_dim)
        self.lin_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        hg = global_mean_pool(h, batch)        # [B, hidden_dim]
        mu = self.lin_mu(hg)                   # [B, latent_dim]
        logvar = self.lin_logvar(hg)           # [B, latent_dim]
        return mu, logvar


class GraphDecoder(nn.Module):
    """
    MLP Decoder: Decode entire graph spin vector [B, N] from graph-level latent z
    (Assumes all graphs have the same size: N = L^2)
    """
    def __init__(self, latent_dim: int, hidden_dim: int, num_nodes: int, out_mode: str = "tanh"):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_nodes)
        assert out_mode in ("tanh", "logits")
        self.out_mode = out_mode

    def forward(self, z, edge_index=None, batch=None):
        # edge_index and batch parameters kept for compatibility with GNN decoder interface
        h = F.relu(self.fc1(z))
        out = self.fc2(h)
        if self.out_mode == "tanh":
            out = torch.tanh(out)  # [-1,1] as spin prediction
        return out                      # [B, N]


class GNNDecoder(nn.Module):
    """
    GNN Decoder: Decode from graph-level latent z, utilizing graph structure
    1. Broadcast z to each node as initial features
    2. Propagate information through GNN layers (neighbor interactions)
    3. Each node outputs a spin
    """
    def __init__(self, latent_dim: int, hidden_dim: int, num_nodes: int, out_mode: str = "tanh"):
        super().__init__()
        self.num_nodes = num_nodes
        assert out_mode in ("tanh", "logits")
        self.out_mode = out_mode
        
        # Map z to node initial features
        self.z_to_node = nn.Linear(latent_dim, hidden_dim)
        
        # GNN layers: enable information propagation between neighbors
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Each node outputs a spin
        self.node_out = nn.Linear(hidden_dim, 1)
    
    def forward(self, z, edge_index, batch):
        """
        z: [B, latent_dim]
        edge_index: [2, E] edges of all graphs (merged in batch)
        batch: [N_total] which graph each node belongs to
        Returns: [B, num_nodes] reconstructed spins
        """
        B = z.size(0)
        
        # 1. Broadcast z to each node
        # First map to hidden_dim
        z_node = self.z_to_node(z)  # [B, hidden_dim]
        
        # Broadcast to all nodes based on batch index
        # batch[i] tells us which graph node i belongs to
        h = z_node[batch]  # [N_total, hidden_dim]
        
        # 2. GNN propagation (neighbor interactions)
        h = F.relu(self.conv1(h, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        
        # 3. Each node outputs a spin
        spin = self.node_out(h)  # [N_total, 1]
        if self.out_mode == "tanh":
            spin = torch.tanh(spin)
        
        # 4. Reshape to [B, num_nodes]
        # Assume each graph has the same number of nodes
        out = spin.view(B, self.num_nodes)
        
        return out


class GNN_VAE_RG(nn.Module):
    """
    GNN-VAE + RG self-supervision:
      - encoder: graph -> (mu, logvar), supports hierarchical RG or simple GNN
      - decoder: z -> spin vector (MLP or GNN)
    """
    def __init__(self, in_dim: int, hidden_dim: int,
                 latent_dim: int, num_nodes: int,
                 use_gnn_decoder: bool = False,
                 use_hierarchical_rg: bool = True,
                 num_rg_steps: int = 3,
                 recon_loss_type: str = "mse"):
        super().__init__()
        self.use_gnn_decoder = use_gnn_decoder
        self.use_hierarchical_rg = use_hierarchical_rg
        assert recon_loss_type in ("mse", "bce")
        self.recon_loss_type = recon_loss_type
        
        # Encoder: hierarchical RG or simple GNN
        if use_hierarchical_rg:
            self.encoder = HierarchicalRGEncoder(in_dim, hidden_dim, latent_dim, num_rg_steps)
        else:
            self.encoder = GNNEncoder(in_dim, hidden_dim, latent_dim)
        
        # Decoder: GNN or MLP
        dec_out_mode = "tanh" if self.recon_loss_type == "mse" else "logits"
        if use_gnn_decoder:
            self.decoder = GNNDecoder(latent_dim, hidden_dim, num_nodes, out_mode=dec_out_mode)
        else:
            self.decoder = GraphDecoder(latent_dim, hidden_dim, num_nodes, out_mode=dec_out_mode)

    def encode(self, data, return_all_levels: bool = False):
        """
        Returns:
          - If return_all_levels=False: (z, mu, logvar) final level
          - If return_all_levels=True: (zs_list, mus_list, logvars_list) all levels
        """
        if return_all_levels and self.use_hierarchical_rg:
            mus, logvars = self.encoder(data, return_all_levels=True)
            zs = []
            for mu, logvar in zip(mus, logvars):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + std * eps
                zs.append(z)
            return zs, mus, logvars
        else:
            mu, logvar = self.encoder(data)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
            return z, mu, logvar

    def decode(self, z, edge_index=None, batch=None):
        if self.use_gnn_decoder:
            return self.decoder(z, edge_index, batch)
        else:
            return self.decoder(z)

    def forward(self, data):
        """
        Returns:
          recon_flat: [B, N]
          mu, logvar, z
        """
        z, mu, logvar = self.encode(data)
        if self.use_gnn_decoder:
            recon_flat = self.decode(z, data.edge_index, data.batch)
        else:
            recon_flat = self.decode(z)
        return recon_flat, mu, logvar, z


# ---------- 2. 2x2 block-spin coarse-graining ----------

def block_spin_2x2(data: Data) -> Data:
    """
    Perform 2x2 block-spin coarse-graining on an LxL Ising graph, get a new L/2 x L/2 graph.
    Assumes data.x: [N,1], N = L^2, spins take values ~{-1,+1}.
    coarse spin = sign(sum of 4 spins) (default to +1 when sum is 0)
    """
    spins = data.x.view(-1)   # [N]
    N = spins.size(0)
    L = int(math.sqrt(N))
    assert L * L == N, "Lattice size must be square"
    assert L % 2 == 0, "L must be even for 2x2 block-spin"

    L2 = L // 2
    spins_2d = spins.view(L, L)

    coarse_spins = torch.empty(L2, L2, dtype=torch.float32)
    for i in range(L2):
        for j in range(L2):
            block = spins_2d[2*i:2*i+2, 2*j:2*j+2]
            s_sum = block.sum()
            s = torch.sign(s_sum)
            if s == 0:
                s = torch.tensor(1.0)   # tie -> +1
            coarse_spins[i, j] = s

    coarse_vec = coarse_spins.view(L2*L2, 1)   # [N2,1]
    edge_index_coarse = build_square_lattice_edge_index(L2, periodic=True)

    coarse_data = Data(
        x=coarse_vec,
        edge_index=edge_index_coarse
    )
    # Preserve original K (for later analysis)
    if hasattr(data, "K"):
        coarse_data.K = data.K.clone()
    return coarse_data


# ---------- 3. Training Loop: VAE + RG Invariance Self-Supervision ----------

def vae_kl_loss(mu, logvar):
    # KL(q(z|x)||N(0,1)) for each sample, shape [B]
    # Return batch-averaged KL
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()


def vae_recon_loss(recon_flat: torch.Tensor, target_spin_pm1: torch.Tensor, recon_loss_type: str) -> torch.Tensor:
    """
    recon_flat: [B, N]
      - if recon_loss_type == "mse": recon_flat is in [-1,1]
      - if recon_loss_type == "bce": recon_flat are logits for p(spin=+1)
    target_spin_pm1: [B, N] in {-1,+1}
    """
    if recon_loss_type == "mse":
        return F.mse_loss(recon_flat, target_spin_pm1)
    elif recon_loss_type == "bce":
        # map spins {-1,+1} -> {0,1} where 1 means +1
        target01 = (target_spin_pm1 + 1.0) * 0.5
        return F.binary_cross_entropy_with_logits(recon_flat, target01)
    else:
        raise ValueError(f"Unknown recon_loss_type={recon_loss_type}")


def train_gnn_vae_rg(
    L: int = 16,
    K_list=None,
    num_samples_per_K: int = 200,
    sweeps_equil: int = 500,
    sweeps_between: int = 10,
    batch_size: int = 32,
    hidden_dim: int = 64,
    latent_dim: int = 4,
    epochs: int = 60,
    beta: float = 0.1,
    lambda_rg: float = 0.5,
    beta_anneal_epochs: int = 0,  # 0 means no annealing; otherwise linearly increase from 0 to beta
    use_mu_for_rg_loss: bool = True,  # RG loss default uses μ (more stable, less sampling noise)
    lr: float = 1e-3,
    save_path: str = "gnn_vae_rg_ising.pt",
    data_path: str = "ising_graph_data.pt",
    force_regenerate: bool = False,
    token_dim: int = 16,
    use_gnn_decoder: bool = False,
    use_hierarchical_rg: bool = True,
    num_rg_steps: int = 2,  # Default changed to 2 levels: 16→8→4
    use_multilevel_rg_loss: bool = True,  # Whether to use multi-level RG loss
    recon_loss_type: str = "mse",  # "mse" or "bce"
):
    if K_list is None:
        K_list = [0.1,0.15,0.2,0.25,0.3,0.32,0.34,0.36,0.38,0.40,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.50,0.52,0.54,0.56,0.58,0.6,0.65,0.7 ]

    # Only use data x/edge/K, not y (phase)
    dataset = IsingGraphDataset(
        L=L,
        K_list=K_list,
        num_samples_per_K=num_samples_per_K,
        sweeps_equil=sweeps_equil,
        sweeps_between=sweeps_between,
        device=device,
        periodic=True,
        data_path=data_path,
        force_regenerate=force_regenerate,
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # tokenizer: encode (s_i, local field) into tokens
    tokenizer = NnInteractionTokenizer(token_dim=token_dim).to(device)

    num_nodes = L * L
    model = GNN_VAE_RG(
        in_dim=token_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_nodes=num_nodes,
        use_gnn_decoder=use_gnn_decoder,
        use_hierarchical_rg=use_hierarchical_rg,
        num_rg_steps=num_rg_steps,
        recon_loss_type=recon_loss_type,
    ).to(device)
    
    encoder_type = f"Hierarchical RG ({num_rg_steps} steps)" if use_hierarchical_rg else "Simple GNN"
    decoder_type = "GNN" if use_gnn_decoder else "MLP"
    print(f"Encoder: {encoder_type}")
    print(f"Decoder: {decoder_type}")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_vae = 0.0
        total_rg = 0.0
        n_samples = 0

        for batch in loader:
            batch = batch.to(device)         # fine graphs
            # Preserve original spins as reconstruction target (don't let tokenizer overwrite)
            B = batch.num_graphs
            target = batch.x.view(B, num_nodes).to(device)  # [B, N]
            B = batch.num_graphs
            n_samples += B

            # First construct coarse graphs based on original spins (block-spin)
            fine_graphs = batch.to_data_list()
            coarse_graphs = [block_spin_2x2(g.cpu()) for g in fine_graphs]
            coarse_batch = Batch.from_data_list(coarse_graphs).to(device)

            # 1) Apply tokenizer to both fine and coarse graphs
            batch = tokenizer(batch)
            coarse_batch = tokenizer(coarse_batch)

            # 2) VAE for fine graph
            if use_multilevel_rg_loss and use_hierarchical_rg:
                # Multi-level mode: get z for all levels
                zs_f, mus_f, logvars_f = model.encode(batch, return_all_levels=True)
                # Use final level z for reconstruction
                z_final = zs_f[-1]
                if model.use_gnn_decoder:
                    recon_flat = model.decode(z_final, batch.edge_index, batch.batch)
                else:
                    recon_flat = model.decode(z_final)
                # Use final level μ, logvar to compute KL
                mu_f = mus_f[-1]
                logvar_f = logvars_f[-1]
            else:
                # Single-level mode
                recon_flat, mu_f, logvar_f, z_f = model(batch)

            # beta annealing (optional)
            if beta_anneal_epochs and beta_anneal_epochs > 0:
                beta_t = beta * min(1.0, epoch / float(beta_anneal_epochs))
            else:
                beta_t = beta

            recon_loss = vae_recon_loss(recon_flat, target, recon_loss_type=recon_loss_type)
            kl = vae_kl_loss(mu_f, logvar_f)
            vae_loss = recon_loss + beta_t * kl

            # 3) RG consistency loss
            if use_multilevel_rg_loss and use_hierarchical_rg:
                # Multi-level RG loss: sum_i ||z^(i) - z^(i+1)||^2
                rg_loss = torch.tensor(0.0, device=device)
                reps = mus_f if use_mu_for_rg_loss else zs_f
                for i in range(len(reps) - 1):
                    rg_loss = rg_loss + F.mse_loss(reps[i], reps[i + 1])
                rg_loss = rg_loss / (len(zs_f) - 1)  # average
            else:
                # Original mode: fine vs coarse batch
                _, mu_c, logvar_c, z_c = model(coarse_batch)
                if use_mu_for_rg_loss:
                    rg_loss = F.mse_loss(mu_f, mu_c)
                else:
                    rg_loss = F.mse_loss(z_f, z_c)

            loss = vae_loss + lambda_rg * rg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            total_vae += vae_loss.item() * B
            total_rg += rg_loss.item() * B

        avg_loss = total_loss / n_samples
        avg_vae = total_vae / n_samples
        avg_rg = total_rg / n_samples
        print(
            f"Epoch {epoch:3d}  total={avg_loss:.4f}  "
            f"VAE={avg_vae:.4f}  RG={avg_rg:.4f}  beta_t={beta_t:.4f}"
        )

    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")


if __name__ == "__main__":
    train_gnn_vae_rg(
        data_path="ising_graph_data.pt",
        force_regenerate=False,
    )

