# config.py
import torch

# --- Model Hyperparameters ---
HIDDEN_CHANNELS = 64

# --- Training Hyperparameters ---
LEARNING_RATE = 0.01
EPOCHS = 100

# --- Environment Settings ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = "."
MODEL_SAVE_PATH = "./saved_models/best_model.pt"