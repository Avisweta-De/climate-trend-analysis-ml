import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(data_path):

    # load dataset
    df = pd.read_csv(data_path)

    # select features
    features = df[["T2M","RH2M","WS10M","PRECTOTCORR"]]

    # model
    model = IsolationForest(contamination=0.01, random_state=42)

    # detect anomalies
    df["ANOMALY"] = model.fit_predict(features)

    return df


