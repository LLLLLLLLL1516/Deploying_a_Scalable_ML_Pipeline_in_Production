# Script to train machine learning model.

# Add the necessary imports for the starter code.
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference
import numpy as np
import pandas as pd
from termcolor import colored, cprint
import pickle
import warnings
warnings.filterwarnings("ignore")


def main():
    # Load in the data.
    data = pd.read_csv("data/census.csv")

    # Train-test split.
    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Process the train data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Train and save a model.
    model = train_model(X_train, y_train)

    # Save the model, encoder, and label binarizer as pickle files
    model_filename = "model/model.pkl"
    lb_filename = "model/label_binarizer.pkl"
    encoder_filename = "model/encoder.pkl"

    with open(model_filename, "wb") as model_file:
        pickle.dump(model, model_file)

    with open(lb_filename, "wb") as lb_file:
        pickle.dump(lb, lb_file)

    with open(encoder_filename, "wb") as encoder_file:
        pickle.dump(encoder, encoder_file)

    # Proces the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    # Make predictions on the test data.
    preds = inference(model, X_test)

    # Compute the model metrics.
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    # Create a dictionary to map metric names to their values and colors
    metrics = {
        "Precision": (precision, "green"),
        "Recall": (recall, "light_blue"),
        "F1": (fbeta, "magenta"),
    }

    # Iterate over the metrics and print them
    for metric, (value, color) in metrics.items():
        cprint(f"{metric}: {value:.2f}", color)

    """
    Output the performance of the model on slices of the data.
    for simplicity, the function can just output the performance on slices of just the categorical features.
    """
    # open a file context to write the metrics
    with open("model/slices_performance.txt", "w") as f:


        # iterate over the categorical features
        for feature in cat_features:
            # slice the data
            unique_values = test[feature].unique()
            for value in unique_values:
                mask = test[feature] == value
                X_slice = test[mask].copy()
                y_slice = y_test[mask].copy()
                # process the data
                X_slice, y_slice, _, _ = process_data(
                    X_slice, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
                )
                # make predictions
                preds_slice = inference(model, X_slice)
                # compute the metrics
                precision, recall, fbeta = compute_model_metrics(y_slice, preds_slice)

                f.write(f"Feature: {feature}, Value: {value}\n")
                f.write(f"Precision: {precision:.2f}\n")
                f.write(f"Recall: {recall:.2f}\n")
                f.write(f"F1: {fbeta:.2f}\n")
                # add a separator
                f.write("-"*20+"\n")

            # close the file context
    f.close()

if __name__ == "__main__":
    main()