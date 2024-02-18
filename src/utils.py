import sys
import os
import numpy as np

import pandas as pd
from src.exception import Custom_Exception
from sklearn.metrics import r2_score,make_scorer
import dill
import pickle
from sklearn.model_selection import GridSearchCV
from src.exception import Custom_Exception


def save_object(file_path,obj):
    try:
            dir_path=os.path.dirname(file_path)
            os.makedirs(dir_path,exist_ok=True)

            with open(file_path,"wb") as file_obj:
                pickle.dump(obj,file_obj)

    except Exception as e:
        raise Custom_Exception(e,sys)     



def evaluate_model(X_train,Y_train,X_test,Y_test,models):
     try:
            report = {}

            for model_name, model in models.items():
                # Fit the model on the training data
                model.fit(X_train, Y_train)
                
                # Predict on both training and test data
                Y_train_pred = model.predict(X_train)
                Y_test_pred = model.predict(X_test)

                # Calculate scores on training and test data
                train_model_score = r2_score(Y_train, Y_train_pred)
                test_model_score = r2_score(Y_test, Y_test_pred)

                # Store the test score in the report dictionary using the model name as the key
                report[model_name] = test_model_score

            return report
     
     except Exception as e:
          raise Custom_Exception(e,sys)
     
    
def load_object(file_path):
       try:
            with(open,"rb") as file_obj:
                 return dill.load(file_path)
            
       except Exception as e:
            raise Custom_Exception(e,sys)


    
