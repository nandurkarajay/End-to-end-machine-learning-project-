import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Build and return the ColumnTransformer preprocessor.
        Numerical features  → median impute → standard scale
        Categorical features → most-frequent impute → one-hot encode
        """
        try:
            numerical_features = ['reading_score', 'writing_score']
            categorical_features = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler',  StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot',  OneHotEncoder(handle_unknown='ignore')),
                ('scaler',  StandardScaler(with_mean=False))
            ])

            logging.info(f'Numerical features   : {numerical_features}')
            logging.info(f'Categorical features : {categorical_features}')

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_features),
                ('cat_pipeline', cat_pipeline, categorical_features)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logging.info('Train and test data read successfully')

            preprocessing_obj = self.get_data_transformer_object()
            logging.info('Preprocessing object obtained')

            target_column_name = 'math_score'

            # Split features / target
            X_train = train_df.drop(columns=[target_column_name])
            y_train = train_df[target_column_name]

            X_test  = test_df.drop(columns=[target_column_name])
            y_test  = test_df[target_column_name]

            # Fit on train, transform both
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr  = preprocessing_obj.transform(X_test)
            logging.info('Preprocessing applied on train and test data')

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr  = np.c_[X_test_arr,  np.array(y_test)]

            # Persist preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Preprocessor object saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
