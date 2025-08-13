# main.py
import os
import torch
import config
from data_loader import get_movielens_data
from model import Model
from train import train_model

def main():
    print('--- Movie Recommender GNN ---')

    # 1) Load data
    data, train_data, val_data, test_data = get_movielens_data(config.DATA_ROOT)

    # 2) Initialize model
    device = config.DEVICE
    model = Model(
        num_users      = data['user'].num_nodes,
        num_movies     = data['movie'].num_nodes,
        metadata       = data.metadata(),
        hidden_channels= config.HIDDEN_CHANNELS,
    ).to(device)

    # 3) Train or load existing
    if os.path.exists(config.MODEL_SAVE_PATH):
        print('✔ Found existing model, loading weights...')
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    else:
        print('❗ No saved model found. Training from scratch...')
        train_model(model, train_data)

    # 4) Evaluate
    print('\n--- Running evaluation ---')
    import evaluate  # expects evaluate.py in same directory
    evaluate.evaluate_model(model, (data, train_data, val_data, test_data))

if __name__ == '__main__':
    main()
