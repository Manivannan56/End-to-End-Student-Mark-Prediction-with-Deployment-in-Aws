import os
import sys
import pandas as pd
from src.exception import Custom_Exception
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import Data_transformation
from src.components.data_transformation import Data_transformation_config

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import Model_trainer


@dataclass
class Data_ingestion_config:
    train_data_path: str=os.path.join('artifacts',"train.csv")  #Used to store the outputs of the data ingestion
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class Data_Ingestion:
    def __init__(self):
        self.ingestion_config=Data_ingestion_config()

    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion method or component")    


        try:
            df=pd.read_csv("/Users/manivannans/Desktop/End_to_End_Machine_learning_project/notebook/stud.csv")
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)                      #Since there is no artifacts folder we are creating it here
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) 

            logging.info("Train test split initiated")    
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Train and test datasets split is completed")

            return(self.ingestion_config.train_data_path, self.ingestion_config.test_data_path,)
        except Exception as e:
           raise Custom_Exception(e,sys)
        

if __name__=="__main__":
    obj=Data_Ingestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation=Data_transformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer=Model_trainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))


