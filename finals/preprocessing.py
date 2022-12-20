#please install catboost to run this code
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV ,train_test_split
from sklearn import metrics
from sklearn.impute import SimpleImputer,KNNImputer
from catboost import CatBoostClassifier
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

#missing=['?']
train_data = pd.read_csv("./data/train_final.csv")#na_values=missing)
test_data= pd.read_csv("./data/test_final.csv")#,na_values=missing)

training_data = pd.read_csv("./data/train_final.csv")#,na_values=missing)
testing_data= pd.read_csv("./data/test_final.csv")#,na_values=missing)
testing_data=testing_data.iloc[: , 1:]
X_train=training_data.iloc[:,:-1]
X_test=testing_data
Y_train=training_data.iloc[:,-1]


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
    return x

def data_preprocessing(X_train,X_test):
    X_train = one_hot_encoding(X_train)
    X_test = one_hot_encoding(X_test)
    X_train = std_scaler(X_train)
    X_test = std_scaler(X_test)
    return X_train,X_test

X_train_processed,X_test=data_preprocessing(X_train,X_test)

def main():
    X_train, X_test, y_train, y_test =train_test_split(X_train_processed, Y_train, test_size=0.20)
    from catboost import CatBoostClassifier
    model = CatBoostClassifier() 
    model_name = 'CATBoost Classification'
    model.fit(X_train, y_train)
    val_pred=model.predict(X_test)
    from sklearn.metrics import accuracy_score
    train_accuracy=accuracy_score(val_pred, y_test)
    Test_accuracy=accuracy_score(val_pred, y_test)

    print("Training accuracy is",train_accuracy)
    print("Validation accuracy is",Test_accuracy)
    Y_sub=model.predict_proba(X_test)[:,1]
    #print(Y_sub)
    pred_list = Y_sub.tolist()
    np.savetxt("cat.csv", pred_list, delimiter=", ", fmt="% s")
    
    
if __name__ == "__main__":
    main()



#____________________-HYpertunung that didnt go well_______________________________
'''
model_eval = evaluate(model, X_train, y_train, X_test, y_test, model_name=model_name)

output(model, X_test_enc_, main,model_name)
parameters ={"depth": sp_randInt(1,14),
             "learning_rate" : sp_randFloat(),
             "iterations" : sp_randInt(100,700)}
model_name = 'CATBoost Classification'
randm=RandomizedSearchCV(estimator=model,param_distributions=parameters, cv =2 ,n_iter =10,n_jobs=-1)
randm.fit(X_train, y_train)
print("Results from random search")
print("best estimator",randm.best_estimator_)
print("best score",randm.best_score_)
print("best parameters",randm.best_params_)
'''
