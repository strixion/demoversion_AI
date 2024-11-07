import os
import pandas as pd


def load_data(data_dir="data"):
    data_frames = {}
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(data_dir, file_name)
            data_frames[file_name] = pd.read_csv(file_path)
            print(f"Загружен файл: {file_name}")
    return data_frames


def preprocess_data(data_frames):
    processed_data = {}
    for name, df in data_frames.items():
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].astype(float)
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                df[column] = df[column].view('int64')
            else:
                df[column] = df[column].astype('category').cat.codes

        processed_data[name] = df
    return processed_data
