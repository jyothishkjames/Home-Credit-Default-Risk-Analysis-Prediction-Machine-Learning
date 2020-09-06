# import libraries
import argparse
import pandas as pd
import numpy as np
import pickle

from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


def load_data(database_filepath):
    """
    Function to load the data from database

    INPUT:
    database_filepath - file path to the database

    OUTPUT:
    X - features
    y - labels
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name='Data_Table', con=engine)
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'], axis=1)
    X = np.array(X)
    y = df['TARGET'].astype(int)
    y = np.array(y)
    return X, y


def build_model():
    """
    Function to build the machine learning pipeline

    OUTPUT:
    model - model that is build
    """
    # build machine learning pipeline
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', RandomForestClassifier())
    ])

    parameters = {
        'scaler__feature_range': [(0, 1)],
        'clf__n_estimators': [10],
        'clf__random_state': [50],
        'clf__class_weight': ['balanced'],
        'clf__max_features': ['auto']  # 'log2', 'sqrt'
    }

    model = GridSearchCV(pipeline, param_grid=parameters)

    return model


def evaluate_model(model, X_test, Y_test):
    """
    Function to evaluate the model

    INPUT:
    model - model to evaluate
    X_test - test feature
    Y_test - test label
    """
    # predict on test data
    Y_pred = model.predict(X_test)
    # display classification report
    print(classification_report(Y_pred, Y_test))
    # display classification accuracy
    accuracy_test = accuracy_score(Y_test, Y_pred)
    print("The model accuracy on test data is: ", accuracy_test)


def save_model(model, model_filepath):
    """
    Function to save the model as a pickle file

    INPUT:
    model - model to save
    model_filepath - path where the model has to saved
    """
    # Save to file the give path
    pkl_filename = model_filepath + "pickle_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


def main():
    # Read the command line arguments and store them
    parser = argparse.ArgumentParser()

    parser.add_argument('--file-path-database', action='store', dest='file_path_database', help='filepath of the '
                                                                                                'database',
                        default=False, required=True)
    parser.add_argument('--file-path-save-model', action='store', dest='file_path', help='filepath to save the model',
                        default=False, required=True)

    results = parser.parse_args()

    print('Loading data...\n    DATABASE: {}'.format(results.file_path_database))
    X, Y = load_data(results.file_path_database)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    print('Building model...')
    model = build_model()

    print('Training model...')
    model.fit(X_train, Y_train)

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test)

    print('Saving model...\n    MODEL: {}'.format(results.file_path))
    save_model(model, results.file_path)

    print('Trained model saved!')


if __name__ == '__main__':
    main()
