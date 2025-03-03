# src/classifier_utils.py

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def train_classifier(X, y, test_size=0.2, random_state=42):
    """
    Train a logistic regression classifier on the provided data.

    Args:
        X (array-like): Feature matrix for training data.
        y (array-like): Labels for training data.
        test_size (float): Proportion of data to be used as the test set (default is 0.2).
        random_state (int): Seed for random number generator for reproducibility (default is 42).

    Returns:
        clf: Trained classifier model.
        X_test: Feature matrix for the test set.
        y_test: Labels for the test set.
    """
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Initialize the logistic regression model
    clf = LogisticRegression(max_iter=1000)
    
    # Fit the classifier to the training data
    clf.fit(X_train, y_train)
    
    # Return the trained model and the test set
    return clf, X_test, y_test

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
