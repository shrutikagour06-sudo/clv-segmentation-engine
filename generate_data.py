import pandas as pd
import numpy as np

np.random.seed(42)
n_rows = 5000

# Generate customer IDs
customer_ids = np.random.choice(range(1001, 1301), size=n_rows)

# Generate transaction dates spanning 1.5 years
start_date = pd.to_datetime('2024-01-01')
end_date = pd.to_datetime('2025-06-30')
date_range_seconds = int((end_date - start_date).total_seconds())
random_seconds = np.random.randint(0, date_range_seconds, size=n_rows)
invoice_dates = start_date + pd.to_timedelta(random_seconds, unit='s')

# Generate realistic retail items, quantities, and prices
quantities = np.random.choice([1, 2, 3, 5, 10, 12, 24], size=n_rows, p=[0.45, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01])
unit_prices = np.random.choice([4.99, 9.99, 14.50, 19.99, 29.99, 49.99, 89.00, 120.00], size=n_rows, p=[0.25, 0.25, 0.15, 0.15, 0.10, 0.05, 0.03, 0.02])

# Generate invoice IDs (grouped logically by date/customer)
df = pd.DataFrame({
    'CustomerID': customer_ids,
    'InvoiceDate': invoice_dates,
    'Quantity': quantities,
    'UnitPrice': unit_prices
}).sort_values('InvoiceDate').reset_index(drop=True)

# Assign invoice numbers based on daily purchasing
df['InvoiceDateStr'] = df['InvoiceDate'].dt.strftime('%Y%m%d')
df['InvoiceNo'] = df.groupby(['CustomerID', 'InvoiceDateStr']).ngroup() + 500000
df = df.drop(columns=['InvoiceDateStr'])

# Save to CSV
output_path = 'sample_retail_transactions.csv'
df[['InvoiceNo', 'CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice']].to_csv(output_path, index=False)
print(f"Generated {len(df)} transactions and saved to '{output_path}'.")