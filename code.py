import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
# %matplotlib inline
import os
print(os.listdir("./data"))
#missing=['?']
data = pd.read_csv("./data/train_final.csv")#,na_values=missing)
submission= pd.read_csv("./data/test_final.csv")#,na_values=missing)
submission=submission.iloc[: , 1:]

print(len(data))
data.head(10)
submission.head(10)
print(data.head(30))
print(data.isnull().sum())
submission.isnull().sum()


data['sex'] = data['sex'].map({'Male': 1, 'Female': 0}) 
submission['sex'] = submission['sex'].map({'Male': 1, 'Female': 0})
#data['race'] = data['race'].map({'White': 1, 'Asian-Pac-Islander': 1, 'Black':0, 'Amer-Indian-Eskimo':0, 'Other':0}) 
#data['relationship'] = data['relationship'].map({'Not-in-family':0, 'Unmarried':0, 'Own-child':0, 'Other-relative':0, 'Husband':1, 'Wife':1})
#data['marital.status'] = data['marital.status'].map({'Widowed':0, 'Divorced':0, 'Separated':0, 'Never-married':0, 'Married-civ-spouse':1, 'Married-AF-spouse':1, 'Married-spouse-absent':0})
#submission['race'] = submission['race'].map({'White': 1, 'Asian-Pac-Islander': 1, 'Black':0, 'Amer-Indian-Eskimo':0, 'Other':0}) 
#submission['relationship'] = submission['relationship'].map({'Not-in-family':0, 'Unmarried':0, 'Own-child':0, 'Other-relative':0, 'Husband':1, 'Wife':1})
#submission['marital.status'] = submission['marital.status'].map({'Widowed':0, 'Divorced':0, 'Separated':0, 'Never-married':0, 'Married-civ-spouse':1, 'Married-AF-spouse':1, 'Married-spouse-absent':0})

labels = ['workclass', 'occupation', 'native.country','education','relationship','marital.status','race']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for l in labels:
    data[l]=le.fit_transform(data[l])
    submission[l]=le.fit_transform(submission[l])

data.head(10)
submission.head(10)

from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(data.loc[:, data.columns != 'income>50K'])
submission = StandardScaler().fit_transform(submission)

Y = data['income>50K']

#________________________________LogReg___________________________________
lst_log = []
classifier = LogisticRegression(max_iter=500)

for train_index, test_index in skf.split(X, Y):
    x_train_fold, x_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = Y[train_index], Y[test_index]
    classifier.fit(x_train_fold, y_train_fold)
    lst_log.append(classifier.score(x_test_fold, y_test_fold))

print('List of possible accuracy:', lst_log)
print('\nMaximum Accuracy That can be obtained from Logistic regression model is:',
      max(lst_log)*100, '%')
#########Hyperparameter tuning for all models ----------
#_____________-hyperparameter tuning log reg___________________
param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet'],
    'C' : [3.5,4.5],
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]
    }
]
clf = GridSearchCV(classifier, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)
best_clf = clf.fit(X,Y)
best_clf.best_estimator_
print (f'Accuracy for hypertuned log regression- : {best_clf.score(X,Y):.3f}')
#exit()
#____'''
#____________________________Bagging__________________________________
bag = BaggingClassifier()
lst_bag = []

for train_index, test_index in skf.split(X, Y):
    x_train_fold, x_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = Y[train_index], Y[test_index]
    bag.fit(x_train_fold, y_train_fold)
    lst_bag.append(bag.score(x_test_fold, y_test_fold))

print('List of possible accuracy:', lst_bag)
print('\nMaximum Accuracy That can be obtained from Bagging is:',
      max(lst_bag)*100, '%')


#________________________________RandomForest__________________________________
rf = RandomForestClassifier()
lst_rf = []

for train_index, test_index in skf.split(X, Y):
    x_train_fold, x_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = Y[train_index], Y[test_index]
    rf.fit(x_train_fold, y_train_fold)
    lst_rf.append(rf.score(x_test_fold, y_test_fold))

print('List of possible accuracy:', lst_rf)
print('\nMaximum Accuracy That can be obtained from RandomForest is:',
      max(lst_rf)*100, '%')


#________________________________Extratree__________________________________

xt = ExtraTreesClassifier()
lst_xt = []

for train_index, test_index in skf.split(X, Y):
    x_train_fold, x_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = Y[train_index], Y[test_index]
    xt.fit(x_train_fold, y_train_fold)
    lst_xt.append(xt.score(x_test_fold, y_test_fold))

print('List of possible accuracy:', lst_xt)
print('\nMaximum Accuracy That can be obtained from Extratree is:',
      max(lst_xt)*100, '%')
#________________________________GradientBoosting__________________________________
gb = GradientBoostingClassifier()
lst_gb = []

for train_index, test_index in skf.split(X, Y):
    x_train_fold, x_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = Y[train_index], Y[test_index]
    gb.fit(x_train_fold, y_train_fold)
    lst_gb.append(gb.score(x_test_fold, y_test_fold))

print('List of possible accuracy:', lst_gb)
print('\nMaximum Accuracy That can be obtained from GradientBoosting is:',
      max(lst_gb)*100, '%')

'''params = {'max_depth': [5, 6, 7], 
         'n_estimators': [100, 150, 200],
          'learning_rate': [0.1, 0.07, 0.05],
          'max_features': ['sqrt', 'log2', 3, 4, 5]
         }

params = {'max_depth': [6], 
         'n_estimators': [200],
          'learning_rate': [0.07, 0.06],
          'max_features': [3,4]
         }


grid = GridSearchCV(gb, param_grid=params, cv=10)
search_result = grid.fit(X, Y)
# GridSearch results
means = search_result.cv_results_['mean_test_score']
params = search_result.cv_results_['params']
for m, p in zip(means, params):
    print(f"{m} with: {p}")

p = np.argmax(means)

best_param = params[p]
print('here')

final_model = GradientBoostingClassifier(best_param)
print('end')

Y_sub=final_model.predict(submission)
Y_sub=Y_sub.reshape((23842,))
pred_list = Y_sub.tolist()
#print(pred_list)
np.savetxt("grad_boosthypertuned_missing.csv", pred_list, delimiter=", ", fmt="% s")
exit()'''


#________________________________Adaboost__________________________________

ada = AdaBoostClassifier()
lst_ada = []

for train_index, test_index in skf.split(X, Y):
    x_train_fold, x_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = Y[train_index], Y[test_index]
    ada.fit(x_train_fold, y_train_fold)
    lst_ada.append(ada.score(x_test_fold, y_test_fold))

print('List of possible accuracy:', lst_ada)
print('\nMaximum Accuracy That can be obtained from Adaboost is:',
      max(lst_ada)*100, '%')

#________________________________LGBM__________________________________
import lightgbm as lgb
 
lil_gbm = lgb.LGBMClassifier(class_weight= 'balanced')
lst_lil_gbm = []

for train_index, test_index in skf.split(X, Y):
    x_train_fold, x_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = Y[train_index], Y[test_index]
    lil_gbm.fit(x_train_fold, y_train_fold)
    lst_lil_gbm.append(lil_gbm.score(x_test_fold, y_test_fold))

print('List of possible accuracy:', lst_lil_gbm)
print('\nMaximum Accuracy That can be obtained from LGBM_ is:',
      max(lst_lil_gbm)*100, '%')


print('trying kfold')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 10)

clf = lgb.LGBMClassifier()#class_weight= 'balanced'
clf.fit(X_test,y_test)
train_pred=clf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(train_pred, y_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, train_pred)))
Y_sub=clf.predict(submission)
Y_sub=Y_sub.reshape((23842,))
pred_list = Y_sub.tolist()
#print(pred_list)
np.savetxt("lgbm_nohypetune_noclass.csv", pred_list, delimiter=", ", fmt="% s")

#_____________-hyperparameter tuning bagging___________________
#_____________-hyperparameter tuning RandomForest___________________
#_____________-hyperparameter tuning_Extratree___________________
#_____________-hyperparameter_tuningGradientBoosting____________________
#_____________-hyperparameter_tuning_Adaboost___________________
#_____________-hyperparameter_tuning_LGBM___________________
#Set the minimum error arbitrarily large