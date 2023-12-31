import logging
from abc import ABC, abstractmethod

import optuna
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


class Model(ABC):
    """
    Abstract base class for all models.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model on the given data.

        Args:
            x_train: Training data
            y_train: Target data
        """
        pass

class LinearRegressionModel(Model):
    def train(self,X_train,y_train,**kwargs):
        """
        Trains the model:
        Args:  
            X_train :Traing data
            y_train: Training labels
        Returns:
            None
        """
        try:
            reg=LinearRegression(**kwargs)
            reg.fit(X_train,y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error("Error in training model : {}".format(e))
            raise e
        