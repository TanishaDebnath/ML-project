import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from data_preprocessing import load_and_split_data

def main(data_path):
    X_train, X_test, y_train, y_test = load_and_split_data(data_path)
    
    with mlflow.start_run():
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(rf, "model")
        print("Logged model and metrics to MLflow.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../data/housing.csv")
    args = parser.parse_args()
    main(args.data_path)
