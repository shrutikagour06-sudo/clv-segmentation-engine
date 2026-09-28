import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.data_pipeline import prepare_rfm
from src.models import cluster_customers, evaluate_clusters

st.set_page_config(page_title="Customer Value & Retention Engine", layout="wide")

st.title("🛒 Customer Lifetime Value (CLV) & Retention Engine")
st.markdown("Automated segmentation pipeline for e-commerce retention decisions.")

# Sidebar Configuration
st.sidebar.header("Configuration")
k_clusters = st.sidebar.slider("Number of Clusters (k)", min_value=3, max_value=6, value=4)
uploaded_file = st.sidebar.file_uploader("Upload Transaction CSV", type=["csv"])

@st.cache_data
def get_sample_data():
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        'InvoiceNo': np.random.randint(100000, 105000, n).astype(str),
        'CustomerID': np.random.randint(1001, 1150, n),
        'InvoiceDate': pd.date_range(start='2025-01-01', periods=n, freq='4h'),
        'Quantity': np.random.randint(1, 10, n),
        'UnitPrice': np.random.uniform(5.0, 100.0, n).round(2)
    })

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
else:
    st.info("💡 Using synthetic demo data. Upload your own CSV via sidebar to analyze real stores.")
    raw_df = get_sample_data()

# Pipeline Execution
with st.spinner("Processing RFM features and running ML models..."):
    rfm_df = prepare_rfm(raw_df)
    results = cluster_customers(rfm_df, n_clusters=k_clusters)

# Top Summary KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", len(results))
col2.metric("Total Revenue", f"${results['Monetary'].sum():,.2f}")
col3.metric("Avg 1-Yr CLV", f"${results['Predicted_1Yr_CLV'].mean():,.2f}")
col4.metric("Top Segment Share", f"{(results['Segment'] == 'VIP / Champions').mean()*100:.1f}%")

# Main Interface Tabs
tab1, tab2 = st.tabs(["📊 Segmentation & Strategy", "🔬 Model Diagnostics (Elbow / Silhouette)"])

with tab1:
    st.subheader("Customer Behavioral Clustering (Interactive 3D)")
    fig = px.scatter_3d(
        results,
        x='Recency',
        y='Frequency',
        z='Monetary',
        color='Segment',
        hover_data=['CustomerID', 'Predicted_1Yr_CLV'],
        title="3D Representation of RFM Segments"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Individual Customer Drill-down Tool
    st.subheader("🔍 Individual Customer Intelligence Lookup")
    all_cust_ids = sorted(results['CustomerID'].unique().tolist())
    selected_cust = st.selectbox("Select Customer ID to Inspect:", all_cust_ids)

    cust_profile = results[results['CustomerID'] == selected_cust].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assigned Segment", cust_profile['Segment'])
    c2.metric("Total Spend", f"${cust_profile['Monetary']:,.2f}")
    c3.metric("Total Orders", int(cust_profile['Frequency']))
    c4.metric("Days Since Last Order", int(cust_profile['Recency']))

    # Actionable Campaign Recommendations
    st.markdown("**Recommended Marketing Action:**")
    if cust_profile['Segment'] == 'VIP / Champions':
        st.success("⭐ **VIP Treatment**: Do not over-discount. Invite to early access drops, concierge support, and premium loyalty perks.")
    elif cust_profile['Segment'] == 'At-Risk / Inactive':
        st.error("⚠️ **Win-back Campaign**: High risk of permanent churn. Trigger a personalized re-engagement discount or feedback survey immediately.")
    elif cust_profile['Segment'] == 'Loyal & Steady':
        st.info("🔄 **Cross-sell / Upsell**: Encourage bundle purchases or category expansion to increase average basket size.")
    else:
        st.warning("🌱 **Nurture Sequence**: Send onboarding emails and highlight popular items to drive their second purchase.")

    st.markdown("---")
    st.subheader("Segment Action Recommendations Table")
    summary = results.groupby('Segment').agg(
        Customer_Count=('CustomerID', 'count'),
        Avg_Recency_Days=('Recency', 'mean'),
        Avg_Spend=('Monetary', 'mean'),
        Avg_Predicted_CLV=('Predicted_1Yr_CLV', 'mean')
    ).round(2).reset_index()

    st.dataframe(summary, use_container_width=True)

    csv_download = results.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Customer Segment CSV", csv_download, "segmented_customers.csv", "text/csv")

with tab2:
    st.subheader("Clustering Validation Metrics")
    st.write("Determine the optimal number of customer groups ($k$) using Inertia and Silhouette Analysis.")
    
    with st.spinner("Calculating cluster quality metrics across k=2 to k=7..."):
        eval_metrics = evaluate_clusters(rfm_df)
    
    m_col1, m_col2 = st.columns(2)
    
    # Elbow Curve Plot
    fig_elbow = px.line(
        eval_metrics, 
        x='k', 
        y='Inertia', 
        markers=True, 
        title="Elbow Method (Look for the 'Bend' / Inertia Drop)"
    )
    m_col1.plotly_chart(fig_elbow, use_container_width=True)

    # Silhouette Score Plot
    fig_sil = px.bar(
        eval_metrics, 
        x='k', 
        y='Silhouette_Score', 
        title="Silhouette Score (Higher is Better Separation)",
        color='Silhouette_Score',
        color_continuous_scale='Viridis'
    )
    m_col2.plotly_chart(fig_sil, use_container_width=True)