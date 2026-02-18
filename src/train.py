import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn

RANDOM_STATE = 42

def make_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    temp = rng.uniform(10, 38, size=n)  # temperatura em °C
    noise = rng.normal(0, 8, size=n)
    sales = 8 * temp + 30 + noise       # relação aproximadamente linear
    sales = np.clip(sales, 0, None)
    return pd.DataFrame({"temperature": temp, "sales": sales})

def main():
    df = make_data()
    X = df[["temperature"]]
    y = df["sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = LinearRegression()

    mlflow.set_experiment("gelato-magico")
    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("mae", float(mae))
        mlflow.log_metric("r2", float(r2))

        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"MAE: {mae:.3f}")
        print(f"R2: {r2:.3f}")

if __name__ == "__main__":
    main()