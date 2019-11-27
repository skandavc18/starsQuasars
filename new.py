import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from Imp import *

data = pd.read_csv('cat1.csv')
print(data.columns[0])
data=data.drop(data.columns[0], axis=1)
data=data.drop(data.columns[1], axis=1)
data=data.drop(data.columns[2], axis=1)
X = data.drop(['pred'], axis=1)
X = X.drop(['class'], axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=20)

red_train=X_train['spectrometric_redshift']
X_train.drop('spectrometric_redshift', axis = 1, inplace = True) #drop redshift else problem is trivial3
red_test=X_test['spectrometric_redshift']
X_test.drop('spectrometric_redshift', axis = 1, inplace = True) #drop redshift else problem is trivial3

#y_train = y_train.drop(0, axis=1).values
#y_test = y_test.drop(0, axis=1).values
X_train=X_train.values
y_train=y_train.values
X_test=X_test.values
y_test=y_test.values

def red_lab(a):
    a=a.to_numpy()
    b=[]
    for i in a:
        if i<=0.0033:
            b.append(1)
        elif i>=0.004:
            b.append(2)
        else:
            b.append(3)
    return b

train_data = Data(X_train, y_train)
eval_data = Data(X_test, y_test)

parameters = {}

print('Start training...')
gbt = GBT()
gbt.train(parameters,
          train_data,
          num_boost_round=5,
          valid_set=eval_data,
          early_stopping_rounds=5)

print('Start predicting...')
y_pred = []
for x in X_test:
    y_pred.append(gbt.predict(x, num_iteration=gbt.best_iteration))

print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
y_pred = [round(x) for x in y_pred]
accuracy=accuracy_score(y_pred,y_test)
print("Accuracy :",accuracy)
red_test=red_lab(red_test)
red_test_2=[]
y_pred_2=[]
for i in range(len(red_test)):
    if red_test[i]!=3:
        y_pred_2.append(y_pred[i])
        red_test_2.append(red_test[i])
print("validation : ",spearmanr(red_test_2,y_pred_2)[0])