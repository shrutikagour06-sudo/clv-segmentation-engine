import pandas as pd
import numpy as np

def prepare_rfm(df: pd.DataFrame, ref_date=None) -> pd.DataFrame:
    df = df.dropna(subset=['CustomerID'])
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalSpend'] = df['Quantity'] * df['UnitPrice']

    if ref_date is None:
        ref_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    else:
        ref_date = pd.to_datetime(ref_date)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (ref_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalSpend': 'sum'
    }).reset_index()

    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    return rfm