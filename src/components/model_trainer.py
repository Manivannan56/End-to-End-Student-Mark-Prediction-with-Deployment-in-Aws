import sys
import os
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import Custom_Exception
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_config_path=os.path.join("artifacts","model.pkl")

class Model_trainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split train and test input data")
            X_train,Y_train,X_test,Y_test=(
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]

            )
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(verbosity=0),
                "Cat Boosting Classifier": CatBoostRegressor(verbose=False),
                "Ada Boosting Classifier": AdaBoostRegressor(),
            }

            model_report:dict=evaluate_model(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,
                                             models=models)
            

            best_scores=max(sorted(model_report.values()))

            best_model_name = max(model_report, key=model_report.get)  # This gets the name of the model with the best score directly
            best_model = models[best_model_name]


            if best_scores < 0.6:
                raise Custom_Exception("No best model found")
            
            logging.info("Best model found on training and test dataset")

            save_object(file_path=self.model_trainer_config.trained_model_config_path,
                        obj=best_model)
            
            predicted=best_model.predict(X_test)
            r2_square=r2_score(Y_test,predicted)
            return r2_square
        except Exception as e:
            raise Custom_Exception(e,sys)
            
