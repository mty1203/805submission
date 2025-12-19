# ising_graph_dataset.py
import os
import torch
from torch.utils.data import Dataset
from typing import Optional, List

from torch_geometric.data import Data

from concurrent.futures import ThreadPoolExecutor, as_completed


class IsingLatticeMC(object):
    """
    2D nearest-neighbor Ising model, K = J/T.
    Spins s_ij = +/-1, LxL periodic boundary conditions.
    """

    def __init__(self, L: int, K: float, device: str = "cpu"):
        self.L = L
        self.K = float(K)
        self.device = device
        # Randomly initialize spins
        self.spins = (
            torch.randint(0, 2, (L, L), device=device, dtype=torch.float32) * 2 - 1
        )  # in {-1, +1}

    def _local_energy(
        self, i: int, j: int, spins: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute local energy contribution of spin at (i,j).
        """
        if spins is None:
            spins = self.spins
        L = self.L
        s = spins

        si = s[i, j]
        # Nearest neighbors
        nn = [
            s[(i + 1) % L, j],
            s[(i - 1) % L, j],
            s[i, (j + 1) % L],
            s[i, (j - 1) % L],
        ]
        E = -self.K * si * sum(nn)
        return E

    def sweep(self, beta: float = 1.0) -> None:
        """
        One Metropolis sweep.
        beta is typically 1 here, K already plays the role of J/T.
        """
        L = self.L
        for _ in range(L * L):
            i = torch.randint(0, L, (1,), device=self.device).item()
            j = torch.randint(0, L, (1,), device=self.device).item()

            old_E = self._local_energy(i, j)
            old_s = self.spins[i, j].clone()
            # Propose flip
            self.spins[i, j] = -old_s
            new_E = self._local_energy(i, j)
            dE = new_E - old_E

            if dE > 0:
                if torch.rand(1, device=self.device) > torch.exp(-beta * dE):
                    self.spins[i, j] = old_s

    @torch.no_grad()
    def get_spin_vector(self) -> torch.Tensor:
        """
        Return entire LxL spin configuration as vector [N], N=L*L.
        """
        L = self.L
        return self.spins.view(L * L)


def build_square_lattice_edge_index(L: int, periodic: bool = True) -> torch.Tensor:
    """
    Construct edge_index for LxL square lattice, shape = [2, num_edges*2] (undirected graph).
    Node numbering: i*L + j, i,j in [0, L-1].
    If periodic=True, use periodic boundary conditions.
    """
    edges = []
    for i in range(L):
        for j in range(L):
            idx = i * L + j

            # Right neighbor
            if j + 1 < L:
                j2 = j + 1
                idx2 = i * L + j2
                edges.append((idx, idx2))
            elif periodic:
                j2 = 0
                idx2 = i * L + j2
                edges.append((idx, idx2))

            # Bottom neighbor
            if i + 1 < L:
                i2 = i + 1
                idx2 = i2 * L + j
                edges.append((idx, idx2))
            elif periodic:
                i2 = 0
                idx2 = i2 * L + j
                edges.append((idx, idx2))

    # Undirected graph: add both directions
    edge_index_list = []
    for u, v in edges:
        edge_index_list.append([u, v])
        edge_index_list.append([v, u])
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()  # [2, E]
    return edge_index


class IsingGraphDataset(Dataset):
    """
    Graph dataset generated from Ising MC simulation.
    Each sample is an LxL lattice -> graph:
      x: [N,1] node features, spins s=+/-1
      edge_index: [2, E] nearest-neighbor edges
      y: [1] phase label (0=high T paramag, 1=low T ferro)
      K: [1] scalar (J/T)
    """

    def __init__(
        self,
        L: int = 16,
        K_list: Optional[List[float]] = None,
        num_samples_per_K: int = 100,
        sweeps_equil: int = 500,
        sweeps_between: int = 10,
        device: str = "cpu",
        periodic: bool = True,
        num_workers: Optional[int] = None,
        data_path: Optional[str] = None,
        force_regenerate: bool = False,
    ):
        """
        num_workers: How many threads to parallelize different K values.
                     Default None uses the smaller of os.cpu_count() or len(K_list).
        data_path: Data save/load path. If provided and file exists and force_regenerate=False, load from file.
        force_regenerate: If True, regenerate data even if data_path exists.
        """
        super().__init__()
        if K_list is None:
            # Around 2D Ising Kc ~ 0.4407
            # K_list = [0.2, 0.3, 0.4, 0.45, 0.5, 0.6]
            K_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7] 

        self.L = L
        self.device = device
        self.edge_index = build_square_lattice_edge_index(L, periodic=periodic)
        self.data_path = data_path

        self.graphs: List[Data] = []

        # Try loading data from file
        if data_path is not None and not force_regenerate and os.path.exists(data_path):
            print(f"[IsingGraphDataset] Loading data from {data_path}")
            self._load_from_file(data_path)
            print(f"[IsingGraphDataset] Loaded {len(self.graphs)} graphs from file")
            return

        # Generate new data
        if num_workers is None:
            num_workers = min(os.cpu_count() or 1, len(K_list))
        print(f"[IsingGraphDataset] Using {num_workers} threads for data generation.")

        # --- Define a worker function to generate sample list for a single K --- #
        def _generate_for_single_K(K_value: float) -> List[Data]:
            # Note: each thread creates its own MC object
            print(f"[IsingGraphDataset] Sampling K={K_value:.3f}")
            mc = IsingLatticeMC(L=L, K=K_value, device=device)

            # Thermalization
            for _ in range(sweeps_equil):
                mc.sweep()

            samples: List[Data] = []
            Kc = 0.4407
            phase_label = 0 if K_value < Kc else 1

            # Sample num_samples_per_K configurations
            for _ in range(num_samples_per_K):
                for _ in range(sweeps_between):
                    mc.sweep()
                spins = mc.get_spin_vector().unsqueeze(-1).cpu()  # [N,1]

                data = Data(
                    x=spins,                  # [N,1]
                    edge_index=self.edge_index,  # Read shared edge_index (not modified)
                    y=torch.tensor([phase_label], dtype=torch.long),
                )
                # Additionally save K information for analysis
                data.K = torch.tensor([K_value], dtype=torch.float32)
                samples.append(data)

            return samples

        # --- Use ThreadPoolExecutor to parallelize different K values --- #
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_generate_for_single_K, K): K for K in K_list
            }
            for future in as_completed(futures):
                K_val = futures[future]
                try:
                    samples_for_K = future.result()
                    self.graphs.extend(samples_for_K)
                except Exception as e:
                    print(f"[IsingGraphDataset] Error when sampling K={K_val}: {e}")

        print(f"[IsingGraphDataset] Total graphs: {len(self.graphs)}")

        # Save data to file
        if data_path is not None:
            self.save_to_file(data_path)
            print(f"[IsingGraphDataset] Saved data to {data_path}")

    def save_to_file(self, path: str) -> None:
        """Save dataset to file"""
        save_dict = {
            'graphs': self.graphs,
            'L': self.L,
            'edge_index': self.edge_index,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(save_dict, path)

    def _load_from_file(self, path: str) -> None:
        """Load dataset from file"""
        save_dict = torch.load(path, map_location='cpu', weights_only=False)
        self.graphs = save_dict['graphs']
        self.L = save_dict['L']
        self.edge_index = save_dict['edge_index']

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]

