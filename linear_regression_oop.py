import pickle as pkl

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split


class RidgeRegression:
    def __init__(self, train, test, alpha, model_path):
        self.train = train
        self.test = test
        self.alpha = alpha
        self.model_path = model_path

    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.train, self.test, test_size=0.2, random_state=0, shuffle=True
        )
        model = Ridge(alpha=self.alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_pred, y_test)
        mape = mean_absolute_percentage_error(y_pred, y_test)
        print(f"Mean square error is {mse}")
        print(f"Mean absolute error is {mape}")
        self.save_model(model, self.model_path)
        return model

    @staticmethod
    def save_model(path_save, model):
        with open(path_save, "wb") as f:
            pkl.dump(model, f)

    @staticmethod
    def load_model(path_to_model):
        with open(path_to_model, "rb") as f:
            model = pkl.load(f)
        return model

    @staticmethod
    def predict(model, x_test):
        y_pred = model.predict(x_test)
        return y_pred

    # def finding_alpha(self,best_alpha=0,mape_choose=0):
    #     minimum_error = float('Inf')
    #     for alpha in range(0,10,0.1):
    #         ridge = Ridge(alpha=alpha)
    #         ridge.fit(self.X_train, self.y_train)
    #         y_pred = ridge.predict(self.X_test)
    #         mse = mean_squared_error(y_pred, self.y_test)
    #         mape = mean_absolute_percentage_error(y_pred, self.y_test)
    #         if mse < minimum_error:
    #         minimum_error = mse
    #         best_alpha = alpha
    #         mape_choose = mape
    #
    #     return best_alpha
