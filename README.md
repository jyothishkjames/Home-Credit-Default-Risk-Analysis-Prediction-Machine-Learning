# Home-Credit-Default-Risk-Prediction

Getting Started
---------------

The Home Credit Default Risk Prediction project is written in python programming language.
The project uses Data Analysis and Machine Learning to predict how capable an applicant is 
of repaying  a loan.

Prerequisites
-------------
The following libraries are used for the project:

        scikit-learn==0.20.4
        pandas==0.24.2
        SQLAlchemy==1.3.17
        numpy==1.16.6
        setuptools==4.1.1
        argparse==1.2.1


Running the Code
----------------

Step 1:

Download the dataset from the below link.

        https://www.kaggle.com/c/home-credit-default-risk/data

Step 2:

Go to the folder **data** and run the code **process_data.py**. This code
will read and pre-process the data and store it in a table **Data_Table** 
in the database **Home_Credit_Default_Predict.db**.  


Step 3:

Go to the folder **models** and run the code **train_classifier.py**. This code
will run the prediction model on the data from the database. The test accuracy and classification report can be seen 
as the output result. 


Running the tests
-----------------

For testing, we have to ensure that the database
**Home_Credit_Default_Predict.db** and the dataset **application_train.csv**
are in the **tests** folder.

Then go to the folder **tests** and follow the below command.

        python -m unittest test.TestTrainClassifier
        python -m unittest test.TestProcessData


Summary
-------

The model attains a classification accuracy of **0.918** on the test data. 
The data file used in this case was **application_train.csv** and 
the test data was generated from the same file, with train and test data split
in the ratio 4:1.   






