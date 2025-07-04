import mlflow
import mlflow.sklearn
import pandas as pd

def predict(model_uri, input_data):
    model = mlflow.sklearn.load_model(model_uri)
    df = pd.DataFrame([input_data])
    pred = model.predict(df)
    print(f"Predicted value: {pred[0]}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_uri", type=str, required=True)
    parser.add_argument("--features", nargs="+", type=float, required=True,
                        help="Feature values in order")
    args = parser.parse_args()
    # Example feature order: ['longitude', 'latitude', 'housing_median_age', 'total_rooms', ...]
    # Adjust as per your dataset!
    columns = [
        "longitude", "latitude", "housing_median_age", "total_rooms",
        "total_bedrooms", "population", "households", "median_income"
    ]
    input_dict = dict(zip(columns, args.features))
    predict(args.model_uri, input_dict)
