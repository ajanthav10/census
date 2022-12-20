import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,StandardScaler,MinMaxScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Perceptron
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
#!pip install catboost
from catboost import CatBoostClassifier
#!pip install lightgbm
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier


#missing=['?']
training_data = pd.read_csv("./data/train_final.csv")#,na_values=missing)
testing_data= pd.read_csv("./data/test_final.csv")#,na_values=missing)
testing_data=testing_data.iloc[: , 1:]
X_train=training_data.iloc[:,:-1]
X_test=testing_data

#X_train=X_train.replace("?",np.NaN)
Y_train=training_data.iloc[:,-1]
#X_test=testing_data.replace("?",np.NaN)




def drop_missingvalues(data)->pd.DataFrame:
    '''implementation of handling missing values
    deleting missing values using replacing '?' as NaN
    ip- dataset
    op- dataset with dropped rows for missing values'''
    data=data.replace("?",np.NaN)
    #can be implemented also as 
    #training_data[training_data=='?']=np.NaN
    data=data.dropna()
    print("training data null values",data.isnull().sum())
    print("Replaced '?' as NaN and dropped rows with missing values")
    return data


def most_freq_values(data)->pd.DataFrame:
    '''implementing missing values to be replaced as most frequent value
    ip - dataset
    op - replaced with most freq values'''
    imputer = SimpleImputer(strategy='most_frequent', 
                            missing_values=np.nan)
    imputer = imputer.fit(data[['workclass','occupation','native.country']])
    data[['workclass','occupation','native.country']] = imputer.transform(data[['workclass','occupation','native.country']])
    #print("training data null values",data.isnull().sum())
    print("Replaced '?' as NaN and applied most_frequent values to it")
    return data

def ordinal_encoding(data)->pd.DataFrame:
    '''Returns ordinally encoded categorical variables
    '''
    cat_features = ['workclass', 'education', 'marital.status', 'occupation',
                'relationship', 'race', 'sex', 'native.country']
    df = data.copy()
    data_cat = df[cat_features]
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=404)
    enc.fit(data_cat)
    data_cat = pd.DataFrame(enc.transform(data_cat), columns=cat_features)
    data[data_cat.columns] = data_cat
    print("transformed categorical values to be label encoded")
    print(data.head(10))
    return data

def knnimpute(data):
    imputer= KNNImputer(n_neighbors=10)
    data=imputer.fit_transform(data)
    #data = pd.DataFrame(imp.transform(data), columns=data.columns)
    return data




def one_hot_encoding(df):
    '''Returns ordinally encoded categorical variables
      cat_features = ['workclass', 'education', 'marital.status', 'occupation',
                'relationship', 'race', 'sex', 'native.country']
    df = data.copy()
    data_cat = df[cat_features]
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(data_cat)
    data_cat = pd.DataFrame(enc.transform(data_cat), columns=cat_features)
    data[data_cat.columns] = data_cat
    print("transformed categorical values to be label encoded")
    print(data.head(10))
    ''' 
    numerical_columns = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    categorical_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
    df_transformed = pd.get_dummies(df, columns=categorical_columns)
    df_final = pd.concat([df[numerical_columns], df_transformed], axis=1)
    return df_final



def std_scaler(x):
    scaler = StandardScaler()
    scaler=scaler.fit(x)
    x = scaler.transform(x)
    #print(X)
    #print(X_test.head(10))
    return x

def data_preprocessing(X_train,X_test):
    '''what to do for data preprocessing  not in the order
    1) normalize - std only
    2) feature selection  -- corr only for numerical values and not for categorical 
    3) encoding - one hot and ordinal
    4) missing values - drop, seperate category,knnimputer, most frequent 
    '''
    
    X_train = one_hot_encoding(X_train)
    X_test = one_hot_encoding(X_test)
    X_train = knnimpute(X_train)
    X_test = knnimpute(X_test)
    X_train = std_scaler(X_train)
    X_test = std_scaler(X_test)
    return X_train,X_test


#print(os.listdir("./data"))
# considering '?' as missing values
#missing=['?']
'''
#sns.heatmap(training_data.corr())
heatmap = sns.heatmap(training_data.corr(numeric_only = False))
plt.figure(figsize=(16, 6))
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')'''


'''print("Length of training data",len(training_data))
print("Length of testing_data data",len(testing_data))
print("shape of training data",training_data.shape)
print("shape of testing_data ",testing_data.shape)
print(training_data.head(10))
print(testing_data.head(10))
#prinitng the datatypes
print(training_data.dtypes)
#calculating the null values
print("training data null values",training_data.isnull().sum())
print("testing data null values",testing_data.isnull().sum())
'''#
#importance(X_train,Y_train)

'''FEATURE PROCESSING 
1) HANDLING MISSING VALUES
    i) k nearest neighbor nd Naives bayes support data with missing values -->why and how ?
    ii) two primary ways of handling missing values
        a) deleting missing values
        b) imputing missing values only categorical feature
            -> most frequent value
            -> treating missing value as separate category 
            -> knearest neighbour
2) Encoding
    i) ordinal encoding
    ii) one hot encoding -> created too much cols 
3) Feature correlation
    -> KNeighborsRegressor
4) data regulariser
    -> numerical data normalization  std scaler
'''

X_train_processed,X_test=data_preprocessing(X_train,X_test)

def log_reg(X_train, X_val, y_train, y_val):
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)
    y_pred = logistic_model.predict(X_val)
    accuracy = accuracy_score(y_val,y_pred)*100
    #cm = metrics.confusion_matrix(y_val, y_pred)
    #print(cm)
    print("accuracy of val set using logistic regression",accuracy)
    return accuracy

def random_forest(X_train, X_val, y_train,y_val):
    randforest_model = RandomForestClassifier(n_estimators=400)
    randforest_model=randforest_model.fit(X_train, y_train)
    y_pred = randforest_model.predict(X_val)
    accuracy = accuracy_score(y_val,y_pred)*100
    print("accuracy of val set using random_forest",accuracy)
    return accuracy

def perceptron(X_train, X_val, y_train, y_val):
    perceptron_model = Perceptron()
    perceptron_model.fit(X_train, y_train)
    y_pred = perceptron_model.predict(X_val)
    accuracy = accuracy_score(y_val,y_pred)*100
    print("accuracy of val set using perceptron",accuracy)
    return accuracy

def adaboost(X_train, X_val, y_train, y_val):
    adaboost_model = AdaBoostClassifier(n_estimators=500)
    adaboost_model=adaboost_model.fit(X_train, y_train)
    y_pred = adaboost_model.predict(X_val)
    accuracy = accuracy_score(y_val,y_pred)*100
    print("accuracy of val set using adaboost",accuracy)
    return accuracy

def bagging(X_train, X_val, y_train, y_val):
    bagging_model = BaggingClassifier(n_estimators=300)
    bagging_model=bagging_model.fit(X_train, y_train)
    y_pred = bagging_model.predict(X_val)
    accuracy = accuracy_score(y_val,y_pred)*100
    print("accuracy of val set using bagging",accuracy)
    return accuracy

def decision_tree(X_train, X_val, y_train, y_val):
    model = DecisionTreeClassifier()
    model=model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val,y_pred)*100
    print("accuracy of val set using decision_tree",accuracy)
    return accuracy

def svm(X_train, X_val, y_train, y_val):
    model = SVC()
    model=model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val,y_pred)*100
    print("accuracy of val set using svm",accuracy)
    return accuracy


def linear_reg(X_train, X_val, y_train, y_val):
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred = linear_model.predict(X_val)
    accuracy = accuracy_score(y_val,y_pred)*100
    print("accuracy of val set using linear_reg",accuracy)
    return accuracy

def catboost(X_train, X_val, y_train, y_val):
    model = CatBoostClassifier()
    model=model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val,y_pred)*100
    print("used predict not predict_proba")
    print("accuracy of val set using catboost",accuracy)
    return accuracy

def lgboost(X_train, X_val, y_train, y_val):
    model = LGBMClassifier()
    model=model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val,y_pred)*100
    print("accuracy of val set using lgboost",accuracy)
    return accuracy


def MLP(X_train, X_val, y_train, y_val):
    for n in [5, 10, 25]:
        print('layers',n)
        model = MLPClassifier(hidden_layer_sizes=(n,), max_iter=800)
        model=model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val,y_pred)*100
        print("accuracy of val set using MLP",accuracy)
    return accuracy


def naive(X_train, X_val, y_train, y_val):
    naive_model=GaussianNB()
    naive_model=naive_model.fit(X_train,y_train)
    y_pred = naive_model.predict(X_val)
    accuracy = accuracy_score(y_val,y_pred)*100
    print("accuracy of val set using naive",accuracy)
    return accuracy

def main():

    X_train, X_val, y_train, y_val = train_test_split(X_train_processed, Y_train, test_size=0.3)
    lgboost(X_train, X_val, y_train, y_val)
    naive(X_train, X_val, y_train, y_val) #--> need cat to numeric conversion 
    MLP(X_train, X_val, y_train, y_val)
    catboost(X_train, X_val, y_train, y_val)
    svm(X_train, X_val, y_train, y_val)
    decision_tree(X_train, X_val, y_train, y_val)
    log_reg(X_train, X_val, y_train, y_val)
    bagging(X_train, X_val, y_train, y_val)
    adaboost(X_train, X_val, y_train, y_val)
    perceptron(X_train, X_val, y_train, y_val)
    random_forest(X_train, X_val, y_train,y_val)
    #linear_reg(X_train, X_val, y_train, y_val)


if __name__ == "__main__":
    main()

'''LIST OF MODELS TO TRAIN
-> create train and val set check which is best pick top three or rive and hypertune it
1) decison tree vanilla
2) bagging
3) random forest
4) adaboost
5) logistic regression
6) perceptron
7) linear regression
8) NN
9) SVM
10) naives bayes
11) lgboost
12) catboost
'''