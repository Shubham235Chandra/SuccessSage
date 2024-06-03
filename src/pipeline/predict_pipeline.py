import sys
import os
import pandas as pd
import numpy as np
import dill

from exception import CustomException
from utils import load_object

class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:

    def __init__(self, **kwargs):
        self.gender = kwargs.get('gender')
        self.race_ethnicity = kwargs.get('race_ethnicity')
        self.parental_level_of_education = kwargs.get('parental_level_of_education')
        self.lunch = kwargs.get('lunch')
        self.test_preparation_course = kwargs.get('test_preparation_course')
        self.reading_score = kwargs.get('reading_score')
        self.writing_score = kwargs.get('writing_score')
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score]
            }

            custom_data_df = pd.DataFrame(custom_data_input_dict)
            return custom_data_df
        
        except Exception as e:
            raise CustomException(e, sys)
