import logging

import mlflow
import numpy as np
import pandas as pd
from src.evaluation import MSE, RMSE, R2
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker
from typing import Tuple


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:

    """
    Args:
        model: RegressorMixin
        x_test: pd.DataFrame
        y_test: pd.Series
    Returns:
        r2_score: float
        rmse: float
    """
    try:
        prediction=model.predict(X_test)
        mse_class=MSE()
        mse=mse_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("mse",mse)
        
        r2_class=R2()
        r2=r2_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("R2",r2)
        
        rmse_class=R2()
        rmse=rmse_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("RMSE",rmse)
        return r2,rmse
    except Exception as e:
        logging.error("Error in evaluationg model : {}".format(e))
        raise e
    
    
    
    # try:
    #     # prediction = model.predict(x_test)
    #     # evaluation = Evaluation()
    #     # r2_score = evaluation.r2_score(y_test, prediction)
    #     # mlflow.log_metric("r2_score", r2_score)
    #     # mse = evaluation.mean_squared_error(y_test, prediction)
    #     # mlflow.log_metric("mse", mse)
    #     # rmse = np.sqrt(mse)
    #     # mlflow.log_metric("rmse", rmse)

    #     prediction = model.predict(x_test)

    #     # Using the MSE class for mean squared error calculation
    #     mse_class = MSE()
    #     mse = mse_class.calculate_score(y_test, prediction)
    #     mlflow.log_metric("mse", mse)

    #     # Using the R2Score class for R2 score calculation
    #     r2_class = R2Score()
    #     r2_score = r2_class.calculate_score(y_test, prediction)
    #     mlflow.log_metric("r2_score", r2_score)

    #     # Using the RMSE class for root mean squared error calculation
    #     rmse_class = RMSE()
    #     rmse = rmse_class.calculate_score(y_test, prediction)
    #     mlflow.log_metric("rmse", np.sqrt(rmse))
        
    #     return r2_score, rmse
    # except Exception as e:
    #     logging.error(e)
    #     raise e
