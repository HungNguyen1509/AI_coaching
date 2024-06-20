import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.datasets import load_diabetes

class RidgeRegression:
    def __init__(self, alpha=1.0, test_size=0.3, random_state=42):
        self.alpha = alpha
        self.test_size = test_size
        self.random_state = random_state
        self.model = Ridge(alpha=self.alpha)
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
        mse = mean_squared_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        return mse, r2

    def finding_alpha(self,best_alpha=0,mape_choose=0):
        minimum_error = float('Inf')
        for alpha in range(0, 10, 1):
            ridge = Ridge(alpha=alpha)
            ridge.fit(self.X_train, self.y_train)
            y_pred = ridge.predict(self.X_test)
            mse = mean_squared_error(y_pred, self.y_test)
            mape = mean_absolute_percentage_error(y_pred, self.y_test)
            if mse < minimum_error:
                minimum_error = mse
                best_alpha = alpha
                mape_choose = mape
        return best_alpha

if __name__ == "__main__":
    # Generate synthetic data (for demonstration purposes)
    data = load_diabetes()
    X,y = data.data, data.target

    #interpret model
    ridge_model = RidgeRegression()

    # Load data into the model
    ridge_model.load_data(X, y)

    # Train the model
    ridge_model.train()

    # Make predictions
    ridge_model.predict()

    #finding_alpha
    best_alpha = ridge_model.finding_alpha()

    # Evaluate the model
    mse, r2 = ridge_model.evaluate()



    # Print evaluation metrics
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    print(best_alpha)
