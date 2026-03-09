# Implementation of Data Creation and Analysis
import numpy as np
import pandas as pd

from plots import plot_data

# Load dataset from CSV file and return features and target labels.
def load_data(csv_path: str, prob_column: str, label_column: str):
    df = pd.read_csv(csv_path)

    if prob_column not in df.columns:
        raise ValueError(f"Missing probability column '{prob_column}' in CSV.")
    if label_column not in df.columns:
        raise ValueError(f"Missing label column '{label_column}' in CSV.")

    X = df[prob_column].astype(float).to_numpy()
    t = df[label_column].astype(int).to_numpy()

    # basic validation
    if np.isnan(X).any() or np.isnan(t).any():
        raise ValueError("CSV contains missing values (NaN), which is not allowed.")

    if (X < 0).any() or (X > 1).any():
        raise ValueError(f"'{prob_column}' must be in [0,1].")

    if not np.isin(t, [0, 1]).all():
        raise ValueError(f"'{label_column}' must contain only 0/1 values.")

    return X, t

# Analyse Data set
def analyse_data(X, t, name, do_print=True, do_plot=True):
    df = pd.DataFrame({'X': X, 't': t})

    if do_print:
        print("Summary statistics:")
        print(df.describe())

        print("\nLabel-wise mean of X:")
        print(df.groupby('t')['X'].mean())

    if do_plot:
        plot_data(X, t, name)

# Predict labels from predicted probabilities using a specified threshold
def predict_labels_from_threshold(X, threshold):
    # Convert predicted probabilities to binary labels using a specified threshold calculated from Youden's J statistic.
    return (X >= threshold).astype(int)