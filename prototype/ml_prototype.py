import pandas as pd
import requests
import io
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

USGS_API_URL = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query.csv"
    "?starttime=2000-01-01&endtime=2023-12-31"
    "&minlatitude=26.0&maxlatitude=31.0&minlongitude=80.0&maxlongitude=89.0"
    "&minmagnitude=4.0"
)

def fetch_data(url=USGS_API_URL):
    resp = requests.get(url)
    df = pd.read_csv(io.StringIO(resp.text))
    return df

def preprocess(df: pd.DataFrame):
    # basic features: mag, depth, lat, lon
    df = df.dropna(subset=["mag", "depth"])
    df["risk"] = (df["mag"] >= 5.0).astype(int)  # crude high-risk label
    return df[["mag", "depth", "latitude", "longitude", "risk"]]

def train_model(df: pd.DataFrame):
    X = df[["mag", "depth", "latitude", "longitude"]]
    y = df["risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    lr = LogisticRegression(max_iter=200)
    params = {"C": [0.1, 1.0, 10.0]}
    clf = GridSearchCV(lr, params, cv=3)
    clf.fit(X_train, y_train)
    print("Best C:", clf.best_params_)
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    df_raw = fetch_data()
    df = preprocess(df_raw)
    train_model(df)