"""Data loader module for the Elliptic Bitcoin dataset.

Constructs a PyTorch Geometric Data object with temporal attributes,
node ID mapping for SAR generation, and temporal train/validation/test split masks.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


class EllipticDatasetLoader:
    """Loads and preprocesses the Elliptic Bitcoin dataset into PyTorch Geometric format.

    Attributes:
        raw_dir (Path): Directory path containing raw CSV files.
        processed_dir (Path): Directory path for saving processed artifacts.
        id_map (Dict[int, int]): Mapping from 0-indexed node indices to raw transaction IDs (txId).
        raw_to_idx (Dict[int, int]): Mapping from raw transaction IDs (txId) to 0-indexed node indices.
        data (Optional[Data]): Built or loaded PyTorch Geometric Data instance.
    """

    def __init__(
        self,
        raw_dir: Union[str, Path] = "data/raw",
        processed_dir: Union[str, Path] = "data/processed",
    ) -> None:
        """Initializes the dataset loader with directory paths.

        Args:
            raw_dir: Path to raw dataset CSV files.
            processed_dir: Path to directory for saving processed graphs.
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.id_map: Dict[int, int] = {}
        self.raw_to_idx: Dict[int, int] = {}
        self.data: Optional[Data] = None

    def build_graph(self) -> Data:
        """Loads raw CSV files and constructs the PyTorch Geometric Data object.

        Returns:
            Data: PyTorch Geometric Data object with:
                - x: FloatTensor [N, 165] (transaction features)
                - edge_index: LongTensor [2, E] (directed transaction flow)
                - y: LongTensor [N] (1=illicit, 0=licit, -1=unknown)
                - time: LongTensor [N] (time step 1..49)
        """
        features_path = self.raw_dir / "elliptic_txs_features.csv"
        classes_path = self.raw_dir / "elliptic_txs_classes.csv"
        edgelist_path = self.raw_dir / "elliptic_txs_edgelist.csv"

        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found at: {features_path}")
        if not classes_path.exists():
            raise FileNotFoundError(f"Classes file not found at: {classes_path}")
        if not edgelist_path.exists():
            raise FileNotFoundError(f"Edgelist file not found at: {edgelist_path}")

        print(f"Loading raw CSVs from {self.raw_dir}...")

        # 1. Load Features (no header: col 0 = txId, col 1 = time_step, cols 2..166 = 165 features)
        df_features = pd.read_csv(features_path, header=None)
        raw_tx_ids = df_features[0].astype(int).values
        time_steps = df_features[1].astype(int).values
        feature_matrix = df_features.iloc[:, 2:].values.astype(np.float32)

        # 2. Build contiguous 0-indexed node mapping
        # id_map: index (0..N-1) -> original txId
        # raw_to_idx: original txId -> index (0..N-1)
        self.id_map = {idx: int(tx_id) for idx, tx_id in enumerate(raw_tx_ids)}
        self.raw_to_idx = {int(tx_id): idx for idx, tx_id in enumerate(raw_tx_ids)}

        # 3. Load Classes & map to 1 (illicit), 0 (licit), -1 (unknown)
        df_classes = pd.read_csv(classes_path)
        class_mapping = {
            "1": 1,        # illicit
            "2": 0,        # licit
            "unknown": -1  # unknown / background
        }
        classes_dict = dict(zip(df_classes["txId"].astype(int), df_classes["class"].astype(str)))

        y_list = [class_mapping.get(classes_dict.get(tx_id, "unknown"), -1) for tx_id in raw_tx_ids]
        y_tensor = torch.tensor(y_list, dtype=torch.long)

        # 4. Load Edgelist and remap txId1, txId2 to contiguous 0-indexed node indices
        df_edges = pd.read_csv(edgelist_path)
        src_mapped = df_edges["txId1"].astype(int).map(self.raw_to_idx)
        dst_mapped = df_edges["txId2"].astype(int).map(self.raw_to_idx)

        # Any txId absent from the features CSV maps to NaN; casting NaN to a
        # long tensor silently truncates to garbage indices, so drop those
        # edges explicitly instead.
        valid_mask = src_mapped.notna() & dst_mapped.notna()
        num_dropped = int((~valid_mask).sum())
        if num_dropped > 0:
            print(
                f"Warning: dropping {num_dropped} edge(s) referencing txId(s) "
                f"absent from {features_path.name}."
            )

        src_nodes = src_mapped[valid_mask].astype(int).values
        dst_nodes = dst_mapped[valid_mask].astype(int).values

        edge_index = torch.tensor(np.vstack([src_nodes, dst_nodes]), dtype=torch.long)

        # 5. Assemble PyG Data object
        x_tensor = torch.tensor(feature_matrix, dtype=torch.float)
        time_tensor = torch.tensor(time_steps, dtype=torch.long)

        self.data = Data(
            x=x_tensor,
            edge_index=edge_index,
            y=y_tensor,
            time=time_tensor
        )
        self.data.id_map = self.id_map

        # Attach default temporal split masks
        self.get_split_masks(train_end=34, val_end=40)

        print(f"Graph construction complete: N={self.data.num_nodes:,}, E={self.data.num_edges:,}")
        return self.data

    def get_split_masks(
        self,
        train_end: int = 34,
        val_end: int = 40
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates temporal split masks based strictly on transaction time steps.

        Args:
            train_end: Highest time step included in train split (inclusive, e.g. 1..34).
            val_end: Highest time step included in validation split (inclusive, e.g. 35..40).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - train_mask: Boolean tensor for time <= train_end
                - val_mask: Boolean tensor for train_end < time <= val_end
                - test_mask: Boolean tensor for time > val_end
        """
        if self.data is None:
            raise ValueError("Graph data not loaded. Call build_graph() or load() first.")

        time_tensor = self.data.time
        train_mask = time_tensor <= train_end
        val_mask = (time_tensor > train_end) & (time_tensor <= val_end)
        test_mask = time_tensor > val_end

        # Attach masks to Data instance
        self.data.train_mask = train_mask
        self.data.val_mask = val_mask
        self.data.test_mask = test_mask

        return train_mask, val_mask, test_mask

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Serializes the PyTorch Geometric Data object and ID mapping to disk.

        Args:
            path: Destination file path. Defaults to data/processed/elliptic_graph.pt.

        Returns:
            Path: Resolved path where the graph was saved.
        """
        if self.data is None:
            raise ValueError("Cannot save empty graph. Call build_graph() or load() first.")

        save_path = Path(path) if path is not None else self.processed_dir / "elliptic_graph.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Store id_map inside data object before saving
        self.data.id_map = self.id_map
        torch.save(self.data, save_path)
        print(f"Graph successfully saved to: {save_path.resolve()}")
        return save_path

    def load(self, path: Optional[Union[str, Path]] = None) -> Data:
        """Loads a serialized PyTorch Geometric Data object from disk.

        Args:
            path: Source file path. Defaults to data/processed/elliptic_graph.pt.

        Returns:
            Data: The loaded PyG Data object.
        """
        load_path = Path(path) if path is not None else self.processed_dir / "elliptic_graph.pt"
        if not load_path.exists():
            raise FileNotFoundError(f"Serialized graph file not found at: {load_path}")

        loaded_obj = torch.load(load_path)

        if isinstance(loaded_obj, dict) and "data" in loaded_obj:
            self.data = loaded_obj["data"]
            self.id_map = loaded_obj.get("id_map", getattr(self.data, "id_map", {}))
        else:
            self.data = loaded_obj
            self.id_map = getattr(loaded_obj, "id_map", {})

        # Reconstruct reverse mapping if id_map is populated
        if self.id_map:
            self.raw_to_idx = {tx_id: idx for idx, tx_id in self.id_map.items()}

        print(f"Graph successfully loaded from: {load_path.resolve()}")
        return self.data


if __name__ == "__main__":
    # 1. Initialize Loader
    loader = EllipticDatasetLoader(
        raw_dir="data/raw",
        processed_dir="data/processed"
    )

    # 2. Build Graph
    print("=" * 65)
    print("BUILDING TEMPORAL AML GRAPH FROM RAW ELLIPTIC DATA")
    print("=" * 65)
    data = loader.build_graph()

    # 3. Print N, E, Tensor Shapes
    print("\n" + "-" * 65)
    print("GRAPH SPECIFICATIONS")
    print("-" * 65)
    print(f"Node count (N)     : {data.num_nodes:,}")
    print(f"Edge count (E)     : {data.num_edges:,}")
    print(f"Feature matrix (x) : {tuple(data.x.shape)} (dtype: {data.x.dtype})")
    print(f"Edge index shape   : {tuple(data.edge_index.shape)} (dtype: {data.edge_index.dtype})")
    print(f"Label tensor (y)   : {tuple(data.y.shape)} (dtype: {data.y.dtype})")
    print(f"Time tensor        : {tuple(data.time.shape)} (min={data.time.min().item()}, max={data.time.max().item()})")
    print(f"Stored id_map size : {len(loader.id_map):,} mapped transaction IDs")

    # 4. Class counts (1=illicit, 0=licit, -1=unknown)
    num_illicit = int((data.y == 1).sum().item())
    num_licit = int((data.y == 0).sum().item())
    num_unknown = int((data.y == -1).sum().item())
    total_labeled = num_illicit + num_licit

    print("\n" + "-" * 65)
    print("CLASS DISTRIBUTION")
    print("-" * 65)
    print(f"Illicit nodes (y=1) : {num_illicit:,} ({num_illicit / data.num_nodes * 100:.2f}%)")
    print(f"Licit nodes (y=0)   : {num_licit:,} ({num_licit / data.num_nodes * 100:.2f}%)")
    print(f"Unknown nodes (y=-1): {num_unknown:,} ({num_unknown / data.num_nodes * 100:.2f}%)")
    print(f"Total labeled nodes : {total_labeled:,} ({total_labeled / data.num_nodes * 100:.2f}%)")
    print(f"Illicit % of labeled: {num_illicit / total_labeled * 100:.2f}%")

    # 5. Split sizes based on data.time
    train_mask, val_mask, test_mask = loader.get_split_masks(train_end=34, val_end=40)
    print("\n" + "-" * 65)
    print("TEMPORAL SPLIT SIZES (Chronological Split)")
    print("-" * 65)
    print(f"Train split (time <= 34)       : {train_mask.sum().item():,} nodes ({train_mask.sum().item() / data.num_nodes * 100:.2f}%)")
    print(f"Validation split (35 <= t <= 40): {val_mask.sum().item():,} nodes ({val_mask.sum().item() / data.num_nodes * 100:.2f}%)")
    print(f"Test split (time > 40)         : {test_mask.sum().item():,} nodes ({test_mask.sum().item() / data.num_nodes * 100:.2f}%)")

    # Labeled nodes per split
    train_labeled = int(((data.y != -1) & train_mask).sum().item())
    val_labeled = int(((data.y != -1) & val_mask).sum().item())
    test_labeled = int(((data.y != -1) & test_mask).sum().item())
    print(f"  • Train labeled nodes        : {train_labeled:,}")
    print(f"  • Val labeled nodes          : {val_labeled:,}")
    print(f"  • Test labeled nodes         : {test_labeled:,}")

    # 6. Save graph artifact
    print("\n" + "-" * 65)
    print("SERIALIZATION")
    print("-" * 65)
    saved_path = loader.save()

    # Quick reload verification
    test_loader = EllipticDatasetLoader()
    reloaded_data = test_loader.load(saved_path)
    assert reloaded_data.num_nodes == 203769
    assert reloaded_data.num_edges == 234355
    assert len(test_loader.id_map) == 203769
    print("✅ Self-test passed: Serialized graph correctly loaded and verified!")
    print("=" * 65)
