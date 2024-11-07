from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(df, target_column):
    """
    Разделяет данные на обучающую и тестовую выборки.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def scale_data(train, test):
    """
    Масштабирует обучающую и тестовую выборки.
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    return train_scaled, test_scaled
