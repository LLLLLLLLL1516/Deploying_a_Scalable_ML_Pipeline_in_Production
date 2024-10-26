import pytest
from fastapi.testclient import TestClient
from main import app # Import the FastAPI app instance
import json

# Create a TestClient for testing the FastAPI app
client = TestClient(app)

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
    response = client.post("https://deploying-a-scalable-ml-pipeline-in-1ptx.onrender.com/predict", json=sample_data)
    assert response.status_code == 200 # Check that the response status code is 200 (OK)
    print("status code is: ", response.status_code)
    print(response.json()["predictions"])