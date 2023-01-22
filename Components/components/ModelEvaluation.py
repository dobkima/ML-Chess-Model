import pandas as pd
import tensorflow as tf
from keras.models import load_model
from mldesigner import command_component, Input, Output

def load_data(eval_data):
    
    eval = pd.read_csv(eval_data)

    X_eval = eval.drop(columns=["start", "end"])
    y_start_eval = eval[["start"]]
    y_end_eval = eval[["end"]]

    return X_eval, y_start_eval, y_end_eval

def load_model(model_file):
    model = load_model(model_file)
    return model

def evaluate_model(start_model, end_model, X_eval, y_start_eval, y_end_eval, evaluation_save_file):
    start_loss, start_accuracy = start_model.evaluate(X_eval, y_start_eval)
    end_loss, end_accuracy = end_model.evaluate(X_eval, y_end_eval)
    pd.DataFrame({"start_loss": [start_loss], "start_accuracy": [start_accuracy], "end_loss": [end_loss], "end_accuracy": [end_accuracy]}).to_csv(evaluation_save_file)
    
    
# Define the pipeline component
@command_component(
    name="model_evaluation",
    version="1.0.0",
    display_name="Evaluate Models",
    description="Evaluates the accuracy of a model to predict the start and end squares of a chess move.",
    environment="conda.yaml"
)
def model_evaluation(
    eval_data: Input(type="uri_file", description="file containing the evaluation data"),
    start_model_file: Input(type="uri_file", description="file containing the start model"),
    end_model_file: Input(type="uri_file", description="file containing the end model"),
    evaluation_save_file: Output(type="uri_file", description="file containing the evaluation results")
    ):
    X_eval, y_start_eval, y_end_eval = load_data(eval_data)
    start_model = load_model(start_model_file)
    end_model = load_model(end_model_file)
    evaluate_model(start_model, end_model, X_eval, y_start_eval, y_end_eval, evaluation_save_file)

