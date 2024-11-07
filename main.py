import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from src.data_preprocessing import load_data, preprocess_data


def main():
    data_frames = load_data(data_dir="data")
    processed_data = preprocess_data(data_frames)

    combined_data = pd.concat(processed_data.values(), axis=0)

    X = combined_data.drop('sales', axis=1)
    y = combined_data['sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    print("Начало обучения модели...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    print("Оценка модели на тестовых данных...")
    loss = model.evaluate(X_test, y_test)
    print(f"Loss на тестовых данных: {loss}")


if __name__ == "__main__":
    main()
