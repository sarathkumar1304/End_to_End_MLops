import logging
from typing import Tuple,Any,List,Dict
import json
import pandas as pd
from src.data_cleaning import (
    DataCleaning,
    DataDivideStrategy,
    DataPreProcessStrategy,
)
import tempfile
import pickle

from typing_extensions import Annotated
from zenml import step


@step
def clean_df(df : pd.DataFrame)->  Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]]:
    """Data cleaning class which preprocesses the data and divides it into train and test data.

    Args:
        data: pd.DataFrame
    returns :
        X_train: Traning data
        X_test : Testing data
        y_train : Training data
        y_test : testing data
    """
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
    
        logging.info("Data cleaning completed ")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Error in cleaning data : {}".format(e))
        raise e