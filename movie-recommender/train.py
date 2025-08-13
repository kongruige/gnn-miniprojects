import torch
import torch.nn as nn
import config


def train_epoch(model, train_data, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()

    # forward
    h = model(train_data.x_dict, train_data.edge_index_dict)

    # ----- use the supervised edge-label pair --------------------------------
    store          = train_data["user", "rates", "movie"]
    user_idx, mov_idx = store.edge_label_index            # (2, 64 000)
    ratings_true      = store.edge_label                  # (64 000,)
    # ------------------------------------------------------------------------

    ratings_pred = model.decode(h["user"], h["movie"], user_idx, mov_idx)
    loss = loss_fn(ratings_pred, ratings_true)

    loss.backward()
    optimizer.step()
    return float(loss)


def train_model(model, train_data):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.MSELoss()

    print("Starting model training...")
    for epoch in range(1, config.EPOCHS + 1):
        loss = train_epoch(model, train_data, optimizer, loss_fn)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}")

    model.cpu().state_dict()  # just to be sure before saving on some systems
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"\nTraining finished. Model saved to {config.MODEL_SAVE_PATH}")
