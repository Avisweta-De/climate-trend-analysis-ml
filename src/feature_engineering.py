import pandas as pd

# Load dataset
df = pd.read_csv("data/climate_processed.csv")

# Feature engineering

# Temperature range
df["TEMP_RANGE"] = df["T2M_MAX"] - df["T2M_MIN"]

# Temperature anomaly
df["TEMP_ANOMALY"] = df["T2M"] - df["T2M"].mean()

# Moving averages
df["TEMP_MA7"] = df["T2M"].rolling(7).mean()
df["TEMP_MA30"] = df["T2M"].rolling(30).mean()

# Rainfall moving average
df["RAIN_MA7"] = df["PRECTOTCORR"].rolling(7).mean()

# Heatwave detection
df["HEATWAVE"] = (df["T2M_MAX"] >= 40).astype(int)

# Extreme rainfall detection
df["EXTREME_RAIN"] = (df["PRECTOTCORR"] > 50).astype(int)

# Remove NaN rows created by rolling window
df = df.dropna()

# Save new dataset
df.to_csv("data/climate_newfeatures.csv", index=False)

print("Feature engineering completed successfully.")