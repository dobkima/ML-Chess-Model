import os
import json
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient

# Import the components to register
from components.DataPreProcessing import data_preprocessing
from components.DataSplitting import data_splitting
from components.EndSquareModel import end_square_model_training
from components.StartSquareModel import start_square_model_training
from components.ModelEvaluation import model_evaluation

# Register the component to server
try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
    # This will open a browser page for
    credential = InteractiveBrowserCredential()

try:
    ml_client = MLClient.from_config(credential=credential)
except Exception as ex:
    # NOTE: Update following workspace information to contain
    #       your subscription ID, resource group name, and workspace name
    client_config = {
        "subscription_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
        "resource_group": "xxxxxxx",
        "workspace_name": "xxxxxxx",
    }

    # write and reload from config file
    config_path = "../.azureml/config.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as fo:
        fo.write(json.dumps(client_config))
    ml_client = MLClient.from_config(credential=credential, path=config_path)


ml_client.components.create_or_update(data_preprocessing)
ml_client.components.create_or_update(data_splitting)
ml_client.components.create_or_update(end_square_model_training)
ml_client.components.create_or_update(start_square_model_training)
ml_client.components.create_or_update(model_evaluation)
