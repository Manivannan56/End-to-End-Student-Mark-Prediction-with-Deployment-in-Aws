import sys
import numpy as np
import os
from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from src.exception import Custom_Exception
from src.logger import logging
from src.utils import save_object

@dataclass
class Data_transformation_config:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl') 

class Data_transformation:
    def __init__(self):
        self.data_tranformation_config=Data_transformation_config()

    def get_data_transformer_object(self):
      " This function is responsible for data transformation for both numerical and the categorical data"
      try:
        numerical_columns=['writing_score','reading_score']
        categorical_columns=[
           "gender",
           "race_ethnicity",
           "parental_level_of_education",
            "lunch",
            "test_preparation_course"
        ]

        num_pipeline=Pipeline(
           steps=[
              ("imputer",SimpleImputer(strategy="median")),
              ("scalar",StandardScaler())
           ]
        )
        cat_pipeline=Pipeline(
           steps=[
              ("imputer",SimpleImputer(strategy="most_frequent")),
              ("one_hot_encoder",OneHotEncoder()),
              ("scalar",StandardScaler(with_mean=False))

           ]
        )
        logging.info("categorical_columns:{categorical_columns}")
        logging.info("numerical_columns:{numerical_columns}")

        preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

        return preprocessor

      except Exception as e:
        raise Custom_Exception(e,sys)
      


    def initiate_data_transformation(self,train_path,test_path):
        try:
          train_df=pd.read_csv(train_path)
          test_df=pd.read_csv(test_path)

          logging.info("Reading train and test data completed")
          
          logging.info("Obatining preprocessor object")

          preprocessing_obj=self.get_data_transformer_object()
          target_column_name="math_score"
          numerical_columns=["writing_score","reading_score"]

          input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
          target_feature_train_df=train_df[target_column_name]

          input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
          target_feature_test_df=test_df[target_column_name]

          logging.info("Applying preprocessing object on training dataframe and test dataframe")

          input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
          input_feature_test_arr=preprocessing_obj.fit_transform(input_feature_test_df)

          train_arr = np.column_stack((input_feature_train_arr, target_feature_train_df))
          test_arr = np.column_stack((input_feature_test_arr, target_feature_test_df))


          logging.info("Saved preprocessing object")

          save_object(
             file_path=self.data_tranformation_config.preprocessor_obj_file_path,
             obj=preprocessing_obj

          )

          return(
             train_arr,
             test_arr,
             self.data_tranformation_config.preprocessor_obj_file_path,
          )
          

        except Exception as e:
           raise Custom_Exception(e,sys)
           

