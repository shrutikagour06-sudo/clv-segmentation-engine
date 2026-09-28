# 🛒 Customer Value & Retention Intelligence Engine

An end-to-end Machine Learning pipeline and interactive analytics dashboard designed for e-commerce customer retention, behavioral segmentation, and 1-year predictive Customer Lifetime Value (CLV) estimation.

## 🚀 Key Features
- **Automated RFM Pipeline:** Aggregates raw order-level logs (`InvoiceNo`, `CustomerID`, `InvoiceDate`, `Quantity`, `UnitPrice`) into Recency, Frequency, and Monetary features with skew-normalization.
- **Unsupervised Segmentation:** Implements K-Means clustering to identify high-value Champions, Loyal regulars, Potential spenders, and At-Risk customer cohorts.
- **Scientific Validation:** Interactive Elbow method (Inertia) and Silhouette Score analysis to justify optimal hyperparameter selection ($k$).
- **Customer Intelligence Lookup:** Search any individual Customer ID to view their behavior profile and automated marketing campaign recommendations.
- **Exportable Insights:** One-click CSV export of segmented customer lists for CRM and marketing automation.

## 🛠️ Tech Stack
- **Language:** Python
- **ML / Analytics:** Scikit-Learn, Pandas, NumPy
- **Visualizations:** Plotly Express
- **Application & UI:** Streamlit

## 📦 Local Setup & Run

1. Clone repository:
   ```bash
   git clone [https://github.com/](https://github.com/)<YOUR_USERNAME>/clv-segmentation-engine.git
   cd clv-segmentation-engine