"""Wine qualtiy prediction model using random forest algorithm - Scikit Learn Python """

# importing libraries
import pandas as pd

# to divide data set into test and traing data
from sklearn.model_selection import train_test_split

# utilities to preprocess the data
from sklearn import preprocessing

# regression model for prediction
from sklearn.ensemble import RandomForestRegressor

# tools for cross validation
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# metrics for model evaluation
from sklearn.metrics import mean_squared_error, r2_score

# loading a dataset
data = pd.read_csv("data.csv")

# separation of target and predictors

y = data.quality
x = data.drop("quality", axis=1)  # taking all attributes of data set except quality as predictor

# dividing data into train and test data

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 123)

# pipeline with preprocessing and model
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

# declare hyperparameters to tune the model
hyperparameters = {'randomforestregressor__max_features':['auto', 'sqrt', 'log2'],
                   'randomforestregressor__max_depth': [None, 5, 3, 1]}

# cross validation with pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

# fit and tune the model
clf.fit(X_train, Y_train)

# for prediction
y_pred = clf.predict(X_test)

# for evaluation
print("Accuracy of model according to r2 Score : ", r2_score(Y_test, y_pred))
print("mean suquared error :", mean_squared_error(Y_test, y_pred))


