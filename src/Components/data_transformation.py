import sys
import os
import pickle
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.logger import logging
from src.exception import CustomException

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        'artifacts',
        'preprocessor.pkl'
    )


class DataTransformation:

    def __init__(self):

        self.data_transformation_config = DataTransformationConfig()

        try:
            logging.info('Data Transformation class has been initialized')

            numerical_features = [
                'writing_score',
                'reading_score'
            ]

            categorical_features = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            logging.info(
                'Numerical and categorical pipelines created successfully'
            )

            # Combine Both Pipelines
            self.preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_features),
                    ('cat_pipeline', cat_pipeline, categorical_features)
                ]
            )

            logging.info(
                'Preprocessing object created successfully'
            )

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            # Read Train and Test Data
            training_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
         

            logging.info(
                'Train and test data read successfully'
            )

            # Preprocessing Object
            preprocessing_obj = self.preprocessor

            target_column_name = 'math_score'

            # X_train and y_train
            input_feature_train_df = training_data.drop(
                columns=[target_column_name]
            )

            target_feature_train_df = training_data[target_column_name]

            # X_test and y_test
            input_feature_test_df = test_data.drop(
                columns=[target_column_name]
            )

            target_feature_test_df = test_data[target_column_name]

            logging.info(
                'Input features and target features separated'
            )

            # Apply preprocessing on training data
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )

            # Apply preprocessing on test data
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )

            logging.info(
                'Preprocessing applied on train and test data'
            )

            # Combine X and y
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            logging.info(
                'Train and test arrays created successfully'
            )

            # Save preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info(
                'Preprocessing object saved successfully'
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)