import unittest
import os
import numpy
import pandas
import sklearn

import process_data
import train_classifier

cwd = os.getcwd()


class TestTrainClassifier(unittest.TestCase):

    def test_load_data(self):
        self.X, self.y = train_classifier.load_data(cwd + "/Home_Credit_Default_Predict.db")
        self.assertIsInstance(self.X, numpy.ndarray)
        self.assertIsInstance(self.y, numpy.ndarray)

    def test_build_model(self):
        self.model = train_classifier.build_model()
        self.assertIsInstance(self.model, sklearn.model_selection._search.GridSearchCV)


class TestProcessData(unittest.TestCase):

    def test_load_clean_data(self):
        self.df = process_data.load_data(cwd + "/application_train.csv")
        self.assertIsInstance(self.df, pandas.core.frame.DataFrame)

    def test_clean_data(self):
        self.df = process_data.load_data(cwd + "/application_train.csv")
        self.df = process_data.clean_data(self.df)
        self.assertIsInstance(self.df, pandas.core.frame.DataFrame)

    def test_polynomial_features(self):
        self.df = process_data.load_data(cwd + "/application_train.csv")
        self.df = process_data.clean_data(self.df)
        self.df = process_data.polynomial_features(self.df)
        self.assertIsInstance(self.df, pandas.core.frame.DataFrame)

    def test_concat_data_frames(self):
        self.df = process_data.load_data(cwd + "/application_train.csv")
        self.df = process_data.clean_data(self.df)
        self.df_poly = process_data.polynomial_features(self.df)
        self.df_poly = process_data.clean_data(self.df)
        self.df = process_data.concat_data_frames(self.df, self.df_poly)
        self.assertIsInstance(self.df, pandas.core.frame.DataFrame)

    def test_create_dummy_df(self):
        self.df = process_data.load_data(cwd + "/application_train.csv")
        self.categorical_features = [feature for feature in self.df.columns if
                                     self.df[feature].dtypes == 'O']
        self.df_cat = self.df[self.categorical_features]
        self.df = process_data.clean_data(self.df)
        self.df_poly = process_data.polynomial_features(self.df)
        self.df_poly = process_data.clean_data(self.df)
        self.df = process_data.concat_data_frames(self.df, self.df_poly)
        self.df = process_data.create_dummy_df(self.df, self.df_cat, dummy_na=False)
