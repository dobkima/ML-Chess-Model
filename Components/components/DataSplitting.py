import pandas as pd
from sklearn.model_selection import train_test_split

from mldesigner import command_component, Input, Output

def train_test_val_split(input_file, train_file, test_file, val_file, test_size, val_size):
    df = pd.read_csv(input_file)
    train, test = train_test_split(df, test_size=test_size)
    train, val = train_test_split(train, test_size=val_size)
    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)
    val.to_csv(val_file, index=False)



# Define the pipeline component
@command_component(
    name="train_test_eval_split",
    version="1.0.0",
    display_name="Split Data",
    description="Splits the data into train, test and evaluation sets.",
    environment="conda.yaml"
)

def data_splitting(
    input_data: Input(type="uri_file", description="PGN file containing chess games"), 
    train_data: Output(type="uri_file", description="file containing the train data"),
    test_data: Output(type="uri_file", description="file containing the test data"),
    eval_data: Output(type="uri_file", description="file containing the evaluation data"),
    test_size = 0.2,
    val_size = 0.2
    ):
    train_test_val_split(input_data, train_data, test_data, eval_data, test_size, val_size)