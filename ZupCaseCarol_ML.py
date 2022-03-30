## Description: Case study Zup. Employee attrition
## Author: Carolina Cortez

## Import Libraries
import os

# Exploratory Data Analysis tools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score, precision_score, \
    classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Model evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder as le, StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix

# Imbalance, tunning
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

# Testing diferent classifier models
def baseline_models(data=[], verbose=False):
    #List of models to be used
    models=[DecisionTreeClassifier(),LogisticRegression(),
            RandomForestClassifier(),GradientBoostingClassifier()]
    X_train, X_test, y_train, y_test = data[0],data[1],data[2],data[3]
    #Instantiate lists to store each of the models results
    accuracy = []
    f1 = []
    auc = []
    recall = []
    precision = []
    #Run thorugh each of the models to get their performance metrics
    for model in models:
        clf = model
        clf.fit(X_train, y_train)
        test_preds = clf.predict(X_test)
        f1.append(f1_score(y_test, test_preds))
        accuracy.append(accuracy_score(y_test, test_preds))
        auc.append(roc_auc_score(y_test, test_preds))
        recall.append(recall_score(y_test, test_preds))
        precision.append(precision_score(y_test, test_preds))
        #Print the model and its report
        if verbose:
            print('Classification Model: ',model,'\n')
            print(classification_report(y_test, test_preds),'\n')
    #store results in dataframe
    results = pd.DataFrame([f1,auc,accuracy, precision,recall],
                      index= ['f1','auc','accuracy','precision','recall',],
                           columns=['DecisionTree','LogisticRegression','RandomForest','Gradient Boosting'])
    #Change orientation of the dataframe
    return results.transpose()

# Change working directory
os.chdir(r'C:\Users\ccorte01\PycharmProjects\pythonProject')

# Load the data
data = pd.read_excel('CaseZup.xlsx')

# Print the shape of the dataset
data.shape

# Print the information of the dataset
data.info()

# Descriptive analysis of the dataset
data.describe()

## Data Cleaning.

# Place 'Attrition' column in first
cols = list(data)
cols[1], cols[0] = cols[0], cols[1]
data = data.reindex(columns=cols)

# Remove irrelevant columns
for column in data.columns[1:]:
    if data[column].nunique() == 1:  # Remove columns with constant values
        data.drop(column, axis=1, inplace=True)
data.drop('EmployeeNumber', axis=1, inplace=True)  # Remove column EmployeeNumber that is just a number assignment

# Remove duplicate rows
data.drop_duplicates(inplace=True)

# Replace Missing Values with Mode
data.fillna(data.mode().iloc[0], inplace=True)

# Print the column data types
print(data.dtypes)

## Analyzing The Data

# Get total number of employees
total = float(len(data))

## Plot general statistics by feature (column)

# Pie charts by feature (column): categorical features
for column in data.columns[0:]:
    if data[column].dtypes == 'object':
        freq = data[column].value_counts()
        plt.figure()
        plt.pie(freq.values, labels=freq.index, autopct='%1.1f%%' )
        plt.title(column)

# Histogram by feature (column): numerical features
for column in data.columns[0:]:
    if data[column].dtypes == 'int64':
        freq = len(data[column].value_counts())
        if freq >= 10:
            freq = 10
        data.hist(column=column,bins=freq,grid=False,edgecolor='black', align='left')
        plt.xlabel(column)
        plt.ylabel('frequency of occurrence')

## Show the attrition correlation with each feature (column)

# Boxplot for numerical features
for column in data.columns[1:]:
    if data[column].dtypes == 'int64':
        plt.figure()
        sns.boxplot(y=column, x='Attrition', data= data, palette="colorblind")
        plt.title(column)

# Stacked plot for categorical features
for column in data.columns[1:]:
    if data[column].dtypes == 'object':
        att_group = data.groupby([column,'Attrition']).size().unstack()
        att_group.apply(lambda x:x/x.sum(),axis=1).plot(kind='barh', stacked=True)
        plt.xlabel('No Attrition / Attrition ratio')
        plt.xlim([0, 1])
        plt.title(column)

## Prediction of Employee's attrition using Machine Learning

#Transform categorical columns into numerical
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = LabelEncoder().fit_transform(data[column])

#Split the data into feature (independent) 'X' and target (dependent) 'Y' variables
X = data.iloc[:, 1:data.shape[1]]
Y = data.iloc[:, 0]

#Creating list of numerical columns
num_col_names = []
for column in data.columns[1:]:
    if data[column].dtypes == 'int64':
        num_col_names.append(column)

# Split the dataset into 75% Training set and 25% Testing set
# Create final train and test sets stratified according to the target variable.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25,stratify=Y, random_state = 0)

# Plot target variable - unbalanced training data
data_train_u = pd.DataFrame(X_train)
data_train_u['Attrition'] = Y_train
freq = data_train_u['Attrition'].value_counts()
plt.figure()
plt.pie(freq.values, labels=freq.index, autopct='%1.1f%%' )
plt.title('Unbalanced training data')
X_train = X_train.iloc[: , :-1]

## Scaling the data

scaler = MinMaxScaler()
#Fit transform the numerical features in the training dataset to a new dataframe
scaled_numfeats_train = pd.DataFrame(scaler.fit_transform(X_train[num_col_names]),
                                     columns=num_col_names, index=X_train.index)
#Integrate scaled values to the training set
for column in num_col_names:
    X_train[column] = scaled_numfeats_train[column]

#Transform the numerical features inthe training dataset to a new dataframe
scaled_numfeats_test = pd.DataFrame(scaler.transform(X_test[num_col_names]),
                                    columns=num_col_names, index=X_test.index)
#Integrate scaled values to the test set
for column in num_col_names:
    X_test[column] = scaled_numfeats_test[column]

## Balancing the data

# Upsample minority class to handle unbalanced target variable
oversample = SMOTE(random_state=0, sampling_strategy=0.6)
X_train_b, Y_train_b = oversample.fit_resample(X_train, Y_train)

# Plot target variable - balanced training data
data_train_b = pd.DataFrame(X_train_b)
data_train_b['Attrition'] = Y_train_b
freq = data_train_b['Attrition'].value_counts()
plt.figure()
plt.pie(freq.values, labels=freq.index, autopct='%1.1f%%' )
plt.title('Balanced training data with 60% of sampling strategy')
X_train_b = X_train_b.iloc[: , :-1]

results = baseline_models(data=[X_train_b, X_test, Y_train_b, Y_test])
results.sort_values('accuracy',ascending=False)
print(results)

# Show the confusion matrix and accuracy for  the model on the test data
# Classification accuracy is the ratio of correct predictions to total predictions made.

logistic = LogisticRegression()
logistic.fit(X_train_b, Y_train_b)
cm = confusion_matrix(Y_test, logistic.predict(X_test))
TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]

print(cm)
print('Model Testing Accuracy = "{}!"'.format((TP + TN) / (TP + TN + FN + FP)))
print()  # Print a new line

# Feature importance (the positive scores indicate a feature that predicts class 1, whereas the negative scores indicate a feature that predicts class 0)
coef=np.round(logistic.coef_,3)
coef=coef.tolist()
coef = pd.DataFrame({'feature':data.iloc[:, 1:data.shape[1]].columns,'coefficient':coef[0]})
coef = coef.sort_values('coefficient',ascending=False).set_index('feature')
print(coef)
coef.plot.barh()
plt.show()
