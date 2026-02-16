import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('pricing_system.db')

# Query data for a specific product
query = "SELECT * FROM market_data WHERE product_name = 'iPhone 15'"
df_product = pd.read_sql(query, conn)

print(df_product.head())
conn.close()