# src/classifier_utils.py

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def train_classifier(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf, X_test, y_test

def evaluate_classifier(clf, X_test, y_test, task_type_to_label):
    y_pred = clf.predict(X_test)
    label_to_task_type = {v: k for k, v in task_type_to_label.items()}
    print(classification_report(y_test, y_pred, target_names=label_to_task_type.values()))
