from eodhd import APIClient

# https://eodhd.com/cp/dashboard
API_KEY = '675be05ca6e2b9.75502389'
client = APIClient(API_KEY)
df = client.get_historical_data(symbol="AAPL.US", interval="d", results=365)
df.to_csv("../data/raw/EODHD-AAPL_US.csv")
print(df.head(10))
