import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error
import pandas as pd
from data_preprocessing import load_and_split_data

def evaluate(model_uri, data_path):
    X_train, X_test, y_train, y_test = load_and_split_data(data_path)
    model = mlflow.sklearn.load_model(model_uri)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"Evaluation RMSE: {rmse}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_uri", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="../data/housing.csv")
    args = parser.parse_args()
    evaluate(args.model_uri, args.data_path)
