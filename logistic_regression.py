import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification

class LogRegression:
    def __init__(self, test_size=0.2, random_state=0):
        self.test_size = test_size
        self.random_state = random_state
        self.model = LogisticRegression()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    def load_data(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    def evaluate(self):
        accuracy = accuracy_score(self.y_test, self.y_pred)
        conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        class_report = classification_report(self.y_test, self.y_pred)
        return accuracy, conf_matrix, class_report


if __name__ == "__main__":
    # Generate synthetic data (for demonstration purposes)
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=0)

    logistic_model = LogRegression()

    # Load data into the model
    logistic_model.load_data(X, y)

    # Train the model
    logistic_model.train()

    # Make predictions
    logistic_model.predict()

    # Evaluate the model
    accuracy, conf_matrix, class_report = logistic_model.evaluate()

    # Print evaluation metrics
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)