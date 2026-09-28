import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def evaluate_clusters(rfm: pd.DataFrame, max_k: int = 7):
    """Computes Inertia and Silhouette scores to determine optimal k."""
    rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].apply(np.log1p)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(rfm_log)

    k_range = list(range(2, max_k + 1))
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_data)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(scaled_data, labels))

    eval_df = pd.DataFrame({
        'k': k_range,
        'Inertia': inertias,
        'Silhouette_Score': silhouettes
    })
    return eval_df

def cluster_customers(rfm: pd.DataFrame, n_clusters: int = 4):
    """Performs K-Means clustering and estimates baseline 1-year CLV."""
    rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].apply(np.log1p)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(rfm_log)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(scaled_data)

    # Label clusters logically by spend & recency
    cluster_means = rfm.groupby('Cluster')[['Recency', 'Monetary']].mean()
    sorted_by_val = cluster_means.sort_values(by='Monetary', ascending=False).index.tolist()
    
    labels = {
        sorted_by_val[0]: 'VIP / Champions',
        sorted_by_val[1]: 'Loyal & Steady',
        sorted_by_val[-1]: 'At-Risk / Inactive'
    }
    rfm['Segment'] = rfm['Cluster'].map(lambda c: labels.get(c, 'Potential Spenders'))

    # Projected 1-Year CLV estimate
    avg_order_value = rfm['Monetary'] / rfm['Frequency']
    purchase_rate_per_day = rfm['Frequency'] / (rfm['Recency'].replace(0, 1) + 30)
    rfm['Predicted_1Yr_CLV'] = (avg_order_value * purchase_rate_per_day * 365).round(2)

    return rfm