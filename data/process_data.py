# import libraries
import argparse
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import Imputer, PolynomialFeatures


def load_data(train_filepath):
    """
    Function to load the data from csv file into dataframe

    INPUT:
    train_filepath - file path to the csv file

    OUTPUT:
    application_train_df - dataframe where the csv was read into
    """
    # load dataset
    df_application_train = pd.read_csv(train_filepath)

    return df_application_train


def clean_data(df):
    """
    Function to clean the data

    INPUT:
    df - dataframe that has missing values

    OUTPUT:
    filled_numerical_df - dataframe with filled missing values
    """
    # Numerical features
    numerical_features = [feature for feature in df.columns if
                          df[feature].dtypes != 'O']

    # Filling missing data for numerical features
    imputer = Imputer(strategy='median')
    df_filled_numerical = imputer.fit_transform(df[numerical_features])
    df_filled_numerical = pd.DataFrame(data=df_filled_numerical, columns=df[numerical_features].columns.values)

    return df_filled_numerical


def polynomial_features(df):
    """
    Function to generate polynomial features

    INPUT:
    df - dataframe from which we select the features to be transformed

    OUTPUT:
    poly_features - dataframe with polynomial features
    """
    # Features to be transformed
    df_poly_features = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

    # Create the polynomial object with specified degree
    poly_transformer = PolynomialFeatures(degree=3)

    # Train the polynomial features
    poly_transformer.fit(df_poly_features)

    # Transform the features
    poly_features = poly_transformer.transform(df_poly_features)

    df_poly_features = pd.DataFrame(poly_features,
                                 columns=poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                                                             'EXT_SOURCE_3', 'DAYS_BIRTH']))

    # Drop columns
    df_poly_features.drop(columns=['1', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'], axis=1,
                       inplace=True)

    return df_poly_features


def concat_data_frames(df_application_train, df_poly_features):
    """
    Function to concatenate dataframes
    """

    return pd.concat([df_application_train, df_poly_features], axis=1)


def create_dummy_df(df_num, df_cat, dummy_na):
    """
    Function to convert categorical data to numerical data

    INPUT:
    num_df - pandas dataframe with numerical variables
    cat_df - pandas dataframe with categorical variables
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not


    OUTPUT:
    df_num - a new dataframe that has the following characteristics:
            1. dummy columns for each of the categorical columns in cat_df
            2. if dummy_na is True - it also contains dummy columns for the NaN values
            3. Use a prefix of the column name with an underscore (_) for separating
    """

    for col in df_cat.columns:
        try:
            df_num = pd.concat([df_num, pd.get_dummies(df_cat[col], prefix=col,
                                                       prefix_sep='_', drop_first=True,
                                                       dummy_na=dummy_na)], axis=1)
        except:
            continue

    return df_num


def save_data(df, database_filepath):
    """
    Function to save the dataframe to a database

    INPUT:
    df - dataframe to save
    database_filepath - path where the database has to saved
    """

    engine = create_engine('sqlite:///' + database_filepath + 'Home_Credit_Default_Predict.db')
    df.to_sql('Data_Table', engine, index=False)


def main():
    # Read the command line arguments and store them
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-path-data', action='store', dest='file_path', help='filepath of the dataset',
                        default=False, required=True)
    parser.add_argument('--file-path-database', action='store', dest='file_path_database', help='filepath of the '
                                                                                                'database',
                        default=False, required=True)

    results = parser.parse_args()

    print('Loading data...\n    application_train: {}'
          .format(results.file_path))
    df_application_train = load_data(results.file_path)

    print('Cleaning data...')
    df_application_train = clean_data(df_application_train)

    print('Creating polynomial features...')
    df_poly_features = polynomial_features(df_application_train)

    print('Cleaning data...')
    df_poly_features = clean_data(df_poly_features)

    print('Concatenating polynomial features with original dataframe...')
    df_concat = concat_data_frames(df_application_train, df_poly_features)

    print('Creating dummy data for categorical features...')

    # Categorical features
    categorical_features = [feature for feature in df_concat.columns if
                            df_concat[feature].dtypes == 'O']

    df = create_dummy_df(df_concat, df_concat[categorical_features], dummy_na=False)

    print('Saving data...\n    DATABASE: {}'.format(results.file_path_database))
    save_data(df, results.file_path_database)

    print('Cleaned data saved to database!')


if __name__ == '__main__':
    main()
