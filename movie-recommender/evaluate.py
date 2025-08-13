# evaluate.py
"""
Dependencies:
  pip install plotly scikit-learn kaleido pandas
"""
import os
import torch
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import config
from data_loader import get_movielens_data
from model import Model

def evaluate_model(model, data_splits):
    # Unpack
    data, train_data, val_data, test_data = data_splits
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    # Ensure output directory exists
    out_dir = 'saved_figures'
    os.makedirs(out_dir, exist_ok=True)

    # 1) Compute embeddings on the train graph
    with torch.no_grad():
        h = model(train_data.x_dict, train_data.edge_index_dict)
    user_emb  = h['user'].cpu()
    movie_emb = h['movie'].cpu()

    # 2) Prepare test predictions
    store    = test_data['user', 'rates', 'movie']
    u_idx, m_idx = store.edge_label_index
    true_r   = store.edge_label.cpu().numpy()
    pred_r   = (user_emb[u_idx] * movie_emb[m_idx]).sum(dim=-1).numpy()

    # Compute metrics
    mse  = mean_squared_error(true_r, pred_r)
    rmse = np.sqrt(mse)
    print(f"Test MSE: {mse:.4f}, RMSE: {rmse:.4f}")

    # 3) Error distribution histogram
    errors = pred_r - true_r
    fig1 = px.histogram(
        errors,
        nbins=50,
        title='Prediction Error Distribution',
        labels={'value': 'Predicted – True Rating', 'count': 'Count'}
    )
    fig1.write_image(os.path.join(out_dir, 'error_histogram.png'))
    fig1.write_html(os.path.join(out_dir, 'error_histogram.html'))
    fig1.show()

    # 4) True vs Predicted scatter
    fig2 = px.scatter(
        x=true_r,
        y=pred_r,
        title='True vs. Predicted Ratings',
        labels={'x': 'True Rating', 'y': 'Predicted Rating'}
    )
    fig2.add_shape(
        type='line', x0=1, y0=1, x1=5, y1=5,
        line=dict(dash='dash')
    )
    fig2.write_image(os.path.join(out_dir, 'true_vs_predicted.png'))
    fig2.write_html(os.path.join(out_dir, 'true_vs_predicted.html'))
    fig2.show()

    # 5) Example predictions
    print('\nSome example predictions:')
    idxs = np.random.choice(len(true_r), size=8, replace=False)
    for i in idxs:
        u = int(u_idx[i].item())
        m = int(m_idx[i].item())
        print(f"  User {u:4d}, Movie {m:4d} — True {true_r[i]:.1f}, Pred {pred_r[i]:.2f}")

    # 6) PCA + KMeans clustering on user embeddings
    pca      = PCA(n_components=2)
    users_2d = pca.fit_transform(user_emb)
    kmeans   = KMeans(n_clusters=5, random_state=0).fit(users_2d)
    labels   = kmeans.labels_.astype(str)
    df = pd.DataFrame({
        'PC1': users_2d[:, 0],
        'PC2': users_2d[:, 1],
        'cluster': labels
    })

    fig3 = px.scatter(
        df,
        x='PC1',
        y='PC2',
        color='cluster',
        title='User Embedding Clusters (PCA + KMeans)',
        labels={'cluster': 'Cluster'}
    )
    fig3.write_image(os.path.join(out_dir, 'user_clusters.png'))
    fig3.write_html(os.path.join(out_dir, 'user_clusters.html'))
    fig3.show()


if __name__ == '__main__':
    # Load data & model
    splits = get_movielens_data(config.DATA_ROOT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = splits[0]
    model = Model(
        num_users      = data['user'].num_nodes,
        num_movies     = data['movie'].num_nodes,
        metadata       = data.metadata(),
        hidden_channels= config.HIDDEN_CHANNELS
    ).to(device)

    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    evaluate_model(model, splits)
