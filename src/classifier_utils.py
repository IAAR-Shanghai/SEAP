"""
Classification utilities for machine learning tasks.

This module provides utility functions for training and evaluating
neural network classifiers on embedding data. It includes functions
for model training, evaluation and performance reporting.

Author: why
Date: 2024
"""

# Standard library imports
from typing import Tuple, Dict, Any

# Third-party imports
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Constants
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_HIDDEN_LAYERS = (512, 256)
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_L2_ALPHA = 1e-4
DEFAULT_MAX_ITER = 200

def train_classifier(
    embeddings: np.ndarray,
    labels: np.ndarray,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE
) -> Tuple[Pipeline, np.ndarray, np.ndarray]:
    """Train a multilayer perceptron classifier.
    
    Trains a classifier using standardized features and a multilayer perceptron.
    The model uses two hidden layers with ReLU activation and Adam optimizer.
    
    Args:
        embeddings: Feature matrix of shape (n_samples, n_features)
        labels: Label array of shape (n_samples,)
        test_size: Proportion of data to use for testing, range (0, 1)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple containing:
            - Trained Pipeline (with scaler and MLP classifier)
            - Test set feature matrix
            - Test set label array
    """
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    # Build Pipeline: Standardization + MLP
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=DEFAULT_HIDDEN_LAYERS,
            activation="relu",
            solver="adam",
            alpha=DEFAULT_L2_ALPHA,
            learning_rate_init=DEFAULT_LEARNING_RATE,
            max_iter=DEFAULT_MAX_ITER,
            random_state=random_state,
            verbose=True
        ))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline, X_test, y_test

def evaluate_classifier(
    clf: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type_to_label: Dict[str, int]
) -> None:
    """Evaluate classifier performance on test set.
    
    Computes and prints a classification report including precision,
    recall, and F1 scores for each class.
    
    Args:
        clf: Trained classifier Pipeline
        X_test: Test set feature matrix
        y_test: True test set labels
        task_type_to_label: Dictionary mapping task types to integer labels
        
    Prints:
        Classification report showing performance metrics for each class
    """
    # Predict test set labels
    y_pred = clf.predict(X_test)
    
    # Reverse label mapping for readability
    label_to_task_type = {v: k for k, v in task_type_to_label.items()}
    
    # Print classification report with readable task type names
    print(classification_report(
        y_test,
        y_pred,
        target_names=label_to_task_type.values()
    ))
