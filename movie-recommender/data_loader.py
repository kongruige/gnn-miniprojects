# data_loader.py
import torch
from torch_geometric.datasets import MovieLens100K
from torch_geometric.transforms import RandomLinkSplit


def get_movielens_data(root_path: str = "."):
    # ----------------------------------------------------------------------
    # 0. Load raw hetero‐graph with separate base/train and test edges
    # ----------------------------------------------------------------------
    dataset = MovieLens100K(root=root_path)
    data = dataset[0]

    # ----------------------------------------------------------------------
    # 1. Merge the “base” edges (data.rating) and the held‐out test edges
    #    (data.edge_label) into a single graph
    # ----------------------------------------------------------------------
    train_ei = data["user", "rates", "movie"].edge_index       # (2, 80000)
    train_r  = data["user", "rates", "movie"].rating.float()   # (80000,)

    test_ei  = data["user", "rates", "movie"].edge_label_index # (2, 20000)
    test_r   = data["user", "rates", "movie"].edge_label       # (20000,)

    # concatenate indices and ratings
    full_ei = torch.cat([train_ei, test_ei], dim=1)           # (2, 100000)
    full_r  = torch.cat([train_r, test_r], dim=0)             # (100000,)

    # overwrite the graph to have a single edge_index + edge_label
    data["user", "rates", "movie"].edge_index    = full_ei
    data["user", "rates", "movie"].edge_label    = full_r

    # delete the old separate fields
    del data["user", "rates", "movie"].rating
    del data["user", "rates", "movie"].edge_label_index

    # ----------------------------------------------------------------------
    # 2. Split into train / val / test with exact 1–5 labels preserved
    # ----------------------------------------------------------------------
    transform = RandomLinkSplit(
        num_val=0.10,
        num_test=0.10,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=0.0,
        edge_types=[("user", "rates", "movie")],
        rev_edge_types=[("movie", "rated_by", "user")],  # correct reverse name
        key="edge_label",  # use our merged 1–5 ratings
    )
    train_data, val_data, test_data = transform(data)

    # ----------------------------------------------------------------------
    # 3. Give every node a 1-D id feature (for the Embedding layer)
    # ----------------------------------------------------------------------
    for split in (train_data, val_data, test_data):
        split["user"].x  = torch.arange(split["user"].num_nodes,  dtype=torch.long)
        split["movie"].x = torch.arange(split["movie"].num_nodes, dtype=torch.long)

    return data, train_data, val_data, test_data
