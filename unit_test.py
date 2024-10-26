# Import libraries
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference
import os
import pickle

# This fixture loads the census data from a CSV file
@pytest.fixture
def data():
    # Read the census data from the specified path
    return pd.read_csv("data/census.csv")

# This fixture provides a list of categorical features for the model
@pytest.fixture
def cat_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

# This fixture provides train and test data after split
@pytest.fixture
def split_data(data):
    train, test = train_test_split(data, test_size=0.20)
    return train, test

def test_process_data_training(split_data, cat_features):
    # Unpack the split data into training and testing sets
    train, _ = split_data
    
    # Process the training data using the specified categorical features
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    
    # Assert that the number of training samples matches the number of labels
    assert X_train.shape[0] == y_train.shape[0], "The number of samples in X_train should match y_train"
    
    # Assert that the encoder was created successfully
    assert encoder is not None, "Encoder should not be None"
    
    # Assert that the label binarizer was created successfully
    assert lb is not None, "Label binarizer should not be None"

def test_process_data_testing(split_data, cat_features):
    # Unpack split data into training and testing sets
    train, test = split_data

    # Process the training data to obtain the encoder and label binarizer
    _, _, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Process the test data using the encoder and label binarizer from training
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    # Assert that the number of samples in X_test matches the number of labels in y_test
    assert X_test.shape[0] == y_test.shape[0], "The number of samples in X_test should match y_test"

def test_train_model(split_data, cat_features):
    # Unpack the split data to obtain the training set
    train, _ = split_data

    # Process the training data to prepare feature and label sets
    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Train the model using the processed training data
    model = train_model(X_train, y_train)

    # Assert that the model was successfully created
    assert model is not None, "Model should not be None after training"

def test_inference(split_data, cat_features):
    # Unpack split data into training and testing sets
    train, test = split_data

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    model = train_model(X_train, y_train)

    # Process the test data using the encoder and label binarizer from training
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    # Run inference to generate predictions for the test data
    preds = inference(model, X_test)

    # Assert that the number of predictions matches the number of test samples
    assert preds.shape[0] == y_test.shape[0], "The number of predictions should match the number of test samples"

def test_compute_model_metrics(split_data, cat_features):
    train, test = split_data

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    model = train_model(X_train, y_train)

    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    preds = inference(model, X_test)

    # Compute model performance metrics (precision, recall, and F1 score)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    # Assert that the computed metrics are within the valid range [0, 1]
    assert 0 <= precision <= 1, "Precision should be between 0 and 1"
    assert 0 <= recall <= 1, "Recall should be between 0 and 1"
    assert 0 <= fbeta <= 1, "F1 score should be between 0 and 1"

def test_model_save(split_data, cat_features):
    # Unpack split data into training and testing sets
    train, test = split_data

    # Process the training data to prepare features and labels, and obtain encoder and label binarizer
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Train the model using the processed training data
    model = train_model(X_train, y_train)

    # Define file paths for saving the model and label binarizer
    model_filename = "model/model.pkl"
    lb_filename = "model/label_binarizer.pkl"

    # Save the trained model and label binarizer to disk
    with open(model_filename, "wb") as model_file:
        pickle.dump(model, model_file)
    with open(lb_filename, "wb") as lb_file:
        pickle.dump(lb, lb_file)
    
    # Ensure files were created successfully
    assert os.path.exists(model_filename), "Model file not saved"
    assert os.path.exists(lb_filename), "Label binarizer file not saved"

    # Load the model and label binarizer from disk
    with open(model_filename, "rb") as model_file:
        loaded_model = pickle.load(model_file)
    with open(lb_filename, "rb") as lb_file:
        loaded_lb = pickle.load(lb_file)

    # Assert that the loaded objects have the same type as the original ones
    assert isinstance(loaded_model, type(model)), "Loaded model type should match original model type"
    assert isinstance(loaded_lb, type(lb)), "Loaded label binarizer type should match original label binarizer type"

    # Cleanup saved files after test
    os.remove(model_filename)
    os.remove(lb_filename)