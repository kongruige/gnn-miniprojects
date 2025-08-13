import torch.nn as nn
from torch_geometric.nn import SAGEConv, to_hetero


class GNNEncoder(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class Model(nn.Module):
    def __init__(self, num_users, num_movies, metadata, hidden_channels=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, hidden_channels)
        self.movie_emb = nn.Embedding(num_movies, hidden_channels)
        self.gnn = to_hetero(
            GNNEncoder(hidden_channels, hidden_channels),
            metadata=metadata,
            aggr="sum",
        )

    def forward(self, x_dict, edge_index_dict):
        initial = {
            "user": self.user_emb(x_dict["user"]),   # already LongTensors
            "movie": self.movie_emb(x_dict["movie"]),
        }
        return self.gnn(initial, edge_index_dict)

    @staticmethod
    def decode(user_emb, movie_emb, user_idx, movie_idx):
        return (user_emb[user_idx] * movie_emb[movie_idx]).sum(dim=-1)
