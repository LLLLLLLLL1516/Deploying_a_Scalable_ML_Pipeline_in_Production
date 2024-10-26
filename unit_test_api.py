import pytest
from fastapi.testclient import TestClient
from main import app # Import the FastAPI app instance
import json

# Create a TestClient for testing the FastAPI app
client = TestClient(app)

# Test for the root endpoint
def test_get_root():
    response = client.get("/") # Send a GET request to the root endpoint
    assert response.status_code == 200 # Check that the response status code is 200 (OK)
    assert response.json() == {"message": "Welcome to the FastAPI App!"} # Check the response message

# Test case for predicting a salary of less than or equal to 50K
def test_post_predict_below():
    # Sample input data for the prediction
    sample_data = {
            "age": 60,
            "workclass": "Private",
            "fnlgt": 160187,
            "education": "9th",
            "education_num": 5,
            "marital_status": "Divorced",
            "occupation": "Other-service",
            "relationship": "Not-in-family",
            "race": "Black",
            "sex": "Female",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 9,
            "native_country": "Cuba"
        }
    
    # Send a POST request to the /predict endpoint with the sample data
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200 # Check that the response status code is 200 (OK)
    assert response.json()["predictions"] == ["<=50K"] # Validate the prediction result

# Test case for predicting a salary of greater than 50K
def test_post_predict_above():
    sample_data = {
            "age": 31,
            "workclass": "Private",
            "fnlgt": 45781,
            "education": "Masters",
            "education_num": 14,
            "marital_status": "Never-married",
            "occupation": "Prof-specialty",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Female",
            "capital_gain": 14084,
            "capital_loss": 0,
            "hours_per_week": 50,
            "native_country": "United-States"
        }
    
    # Send a POST request to the /predict endpoint with the sample data
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    assert response.json()["predictions"] == [">50K"] # Validate the prediction result