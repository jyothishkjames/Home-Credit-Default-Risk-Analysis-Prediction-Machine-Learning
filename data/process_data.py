# import libraries
import sys

import pandas as pd

from sqlalchemy import create_engine
from sklearn.preprocessing import Imputer


def load_data(train_filepath, test_filepath):
    # load train dataset
    application_train_df = pd.read_csv(train_filepath)

    return application_train_df


def clean_data(application_train_df):
    # Categorical features
    categorical_features = [feature for feature in application_train_df.columns if
                            application_train_df[feature].dtypes == 'O']

    # Numerical features
    numerical_features = [feature for feature in application_train_df.columns if
                          application_train_df[feature].dtypes != 'O']

    # Filling missing data for numerical features
    application_train_relevant_df = application_train_df[numerical_features]
    imputer = Imputer(strategy='median')
    filled_numerical_df = imputer.fit_transform(application_train_relevant_df)
    filled_numerical_df = pd.DataFrame(data=filled_numerical_df, columns=application_train_relevant_df.columns.values)

    return filled_numerical_df, categorical_features


def create_dummy_df(num_df, cat_df, dummy_na):
    """
    INPUT:
    num_df - pandas dataframe with numerical variables
    cat_df - pandas dataframe with categorical variables
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not


    OUTPUT:
    df - a new dataframe that has the following characteristics:
            1. dummy columns for each of the categorical columns in cat_df
            2. if dummy_na is True - it also contains dummy columns for the NaN values
            3. Use a prefix of the column name with an underscore (_) for separating
    """

    for col in cat_df.columns:
        try:
            num_df = pd.concat([num_df, pd.get_dummies(cat_df[col], prefix=col,
                                                       prefix_sep='_', drop_first=True,
                                                       dummy_na=dummy_na)], axis=1)
        except:
            continue

    return num_df


def save_data(df, database_filepath):
    engine = create_engine('sqlite:///Home_Credit_Default_Predict.db')
    df.to_sql('Data_Table', engine, index=False)


def main():
    if len(sys.argv) == 4:

        train_filepath, test_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(train_filepath, test_filepath))
        application_train_df= load_data(train_filepath, test_filepath)

        print('Cleaning data...')
        df, categorical_features = clean_data(application_train_df)

        print('Creating dummy data for categorical features and concatenating it with numerical features...')
        df = create_dummy_df(df, df[categorical_features], dummy_na=False)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the train and test ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'application_train.csv application_test.csv ' \
              'Home_Credit_Default_Predict.db')


if __name__ == '__main__':
    main()
