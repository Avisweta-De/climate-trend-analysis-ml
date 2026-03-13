import pandas as pd

# Load dataset
df = pd.read_csv("../Data/climate_processed.csv")

# Feature engineering
df["TEMP_RANGE"] = df["T2M_MAX"] - df["T2M_MIN"]
df["TEMP_ANOMALY"] = df["T2M"] - df["T2M"].mean()

df["TEMP_MA7"] = df["T2M"].rolling(7).mean()
df["TEMP_MA30"] = df["T2M"].rolling(30).mean()

df["RAIN_MA7"] = df["PRECTOTCORR"].rolling(7).mean()

df["HEATWAVE"] = (df["T2M_MAX"] >= 40).astype(int)
df["EXTREME_RAIN"] = (df["PRECTOTCORR"] > 50).astype(int)

df = df.dropna()

# Save new dataset
df.to_csv("../data/climate_features.csv", index=False)

print("Feature engineering complete.")