import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def load_data_from_csv(file_path, target_column):
    """Loads data from a CSV file and prepares it for KNN."""
    df = pd.read_csv(file_path)
    
    # Separate features (X) and target (y)
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values
    
    return X, y, df

def train_and_evaluate_knn(X, y, n_neighbors=3, test_size=0.2, random_state=42):
    """Trains a KNN classifier and evaluates its accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return knn, accuracy
