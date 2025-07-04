import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(data_path, test_size=0.2, random_state=42):
    df = pd.read_csv(data_path)
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../data/housing.csv")
    args = parser.parse_args()
    X_train, X_test, y_train, y_test = load_and_split_data(args.data_path)
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
