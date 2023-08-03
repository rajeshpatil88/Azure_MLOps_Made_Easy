# Step 1: Load the Trained Model
import pickle
from sklearn.linear_model import LinearRegression

# Sample data for training
X_train = [[1], [2], [3], [4], [5]]
y_train = [2, 4, 6, 8, 10]

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to a .pkl file
with open("linear_regression_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Step 2: Set Up Azure ML Workspace
# Ensure you have an Azure Machine Learning workspace set up.
# If you don't have one, follow the instructions to create one.

# Step 3: Initialize Azure ML Workspace
from azureml.core import Workspace

# Replace 'your_workspace_name' and 'your_subscription_id' with your actual values
ws = Workspace.get(name='ws_1', subscription_id='2152fd0e-4534-43c6-9496-6e6de93b53db',resource_group='Model_deployment_WS')

# Step 4: Create a Scoring Script
# Create a Python script named score.py that defines the init() and run() functions.
# Refer to the Step 4 code provided in the previous response.

# Step 5: Define Environment and Configuration
from azureml.core import Environment
from azureml.core.model import InferenceConfig

# Create a Python environment
env = Environment.from_conda_specification(name="env", file_path="environment.yml")

# Define the inference configuration
inference_config = InferenceConfig(entry_script="score.py", environment=env)

# Step 6: Deploy the Model
from azureml.core.model import Model

# Register the model in the workspace
model = Model.register(workspace=ws, model_name="linear_regression_model", model_path="linear_regression_model.pkl")

# Deploy the model as a web service
service_name = "linear-regression-service"
service = Model.deploy(workspace=ws, name=service_name, models=[model], inference_config=inference_config, deployment_config=None, overwrite=True)
service.wait_for_deployment(show_output=True)

# Step 7: Test the Web Service
# After deployment, get the web service endpoint URL.
endpoint = service.scoring_uri

# Sample input data for prediction
input_data = [[6], [7], [8]]

# Send a batch of data for prediction
import requests
import json

input_data_json = json.dumps({"data": input_data})
headers = {"Content-Type": "application/json"}
response = requests.post(endpoint, input_data_json, headers=headers)
predictions = response.json()
print(predictions)
print(service.get_logs())

# Step 8: Monitor the Web Service
# You can monitor the web service using Azure ML monitoring tools to track its performance and diagnose any potential issues.
