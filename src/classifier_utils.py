# src/classifier_utils.py

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_classifier(embeddings, labels, test_size=0.2, random_state=42):
    """
    Train a multi-layer perceptron classifier on the given embeddings and labels.

    Returns:
        trained pipeline (scaler + MLP), test features, test labels
    """
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Build pipeline with standardization + MLP
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(512, 256),  # Two hidden layers
            activation="relu",
            solver="adam",
            alpha=1e-4,             # L2 regularization
            learning_rate_init=1e-3,
            max_iter=200,
            random_state=random_state,
            verbose=True
        ))
    ])

    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test

def evaluate_classifier(clf, X_test, y_test, task_type_to_label):
    """
    Evaluate the classifier on the test set and print a classification report.

    Args:
        clf: Trained classifier model.
        X_test (array-like): Feature matrix for the test set.
        y_test (array-like): True labels for the test set.
        task_type_to_label (dict): Mapping from task labels to task types.

    Prints:
        A classification report showing precision, recall, F1-score for each class.
    """
    # Predict labels for the test set
    y_pred = clf.predict(X_test)
    
    # Invert the label mapping (task_type_to_label) to get the label names
    label_to_task_type = {v: k for k, v in task_type_to_label.items()}
    
    # Print the classification report with human-readable task type names
    print(classification_report(y_test, y_pred, target_names=label_to_task_type.values()))
