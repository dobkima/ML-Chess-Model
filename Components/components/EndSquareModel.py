import pandas as pd
import tensorflow as tf
from keras.layers import Dense, InputLayer
from keras.models import Sequential
from mldesigner import command_component, Input, Output

def load_data(train_data, test_data):
    train = pd.read_csv(train_data)
    test = pd.read_csv(test_data)

    X_train = train.drop(columns=["start", "end"])
    y_train = train[["end"]]

    X_test = test.drop(columns=["start", "end"])
    y_test = test[["end"]]

    return X_train, y_train, X_test, y_test

def build_model():

    model = Sequential([
        InputLayer(input_shape=(66,)),
        Dense(32, activation="relu"),
        Dense(32, activation="relu"),
        Dense(32, activation="relu"),
        Dense(64, activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs, model_save_file, history_save_file):
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    model.save(model_save_file)
    pd.DataFrame(history.history).to_csv(history_save_file)

# Define the pipeline component
@command_component(
    name="end_square_model_training",
    version="1.0.0",
    display_name="Train Destination Square Model",
    description="Trains a model to predict the destination square of a chess move.",
    environment="conda.yaml"
)
def end_square_model_training(
    train_data: Input(type="uri_file", description="file containing the training data"),
    test_data: Input(type="uri_file", description="file containing the test data"),
    model_save_file: Output(type="uri_file", description="file containing the trained model"),
    history_save_file: Output(type="uri_file", description="file containing the training history"),
    epochs = 10
    ):
    X_train, y_train, X_test, y_test = load_data(train_data, test_data)
    model = build_model()
    train_model(model, X_train, y_train, X_test, y_test, epochs, model_save_file, history_save_file)

