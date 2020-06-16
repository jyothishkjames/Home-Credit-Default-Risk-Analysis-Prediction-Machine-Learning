# import libraries
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb

from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, accuracy_score


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///Home_Credit_Default_Predict.db')
    df = pd.read_sql_table(table_name='Data_Table', con=engine)
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'], axis=1)
    X = np.array(X)
    y = df['TARGET'].astype(int)
    y = np.array(y)
    return X, y


def build_model():
    # build machine learning pipeline
    pipeline = Pipeline([
        ('scale', MinMaxScaler(feature_range=(0, 1))),
        ('clf', lgb.LGBMClassifier(n_estimators=10000, objective='binary',
                                   class_weight='balanced', learning_rate=0.05,
                                   reg_alpha=0.1, reg_lambda=0.1,
                                   subsample=0.8, n_jobs=-1, random_state=50))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    # predict on test data
    Y_pred = model.predict(X_test)
    # Iterating through each column
    for col1, col2 in zip(Y_pred, Y_test):
        print(classification_report(col1, col2))


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/Home_Credit_Default_Predict.db classifier.pkl')


if __name__ == '__main__':
    main()
