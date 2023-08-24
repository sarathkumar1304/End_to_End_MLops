import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class Evaluation(ABC):
    """
    Abstract Class defining the strategy for evaluating model performance
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass


class MSE(Evaluation):
    """
    Evaluation strategy that uses Mean Squared Error (MSE)
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            mse: float
        """
        try:
            logging.info("calaculating  MSE ")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE : {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calaculating MSE : {}".format(e))
            raise e


class R2(Evaluation):
    """
    Evaluation strategy that uses R2 Score
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            r2_score: float
        """
        try:
            logging.info("Calculatimg R2_Score")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2_score :{}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calaculating R2 score".format(e))
            raise e


class RMSE(Evaluation):
    """
    Evaluation strategy that uses Root Mean Squared Error (RMSE)
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            rmse: float
        """
        try:
            logging.info(" calculating RMSE value")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info("RMSE : {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calaculating RMSE:{}".format(e))
            raise e
