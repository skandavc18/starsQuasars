import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, roc_auc_score, roc_curve
from scipy.stats import spearmanr
from bias_variance_decomp1 import *
data = pd.read_csv('../cat4.csv')


data.drop(data.columns[0], axis = 1, inplace = True) 
data.drop(data.columns[1],axis = 1, inplace = True)
data.drop(data.columns[0],axis = 1, inplace = True)
data.drop(['pred'], axis=1,inplace=True)


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

X= data.drop(['class'], axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=20)

red_train=X_train['spectrometric_redshift']
X_train.drop('spectrometric_redshift',axis=1,inplace=True)
red_test=X_test['spectrometric_redshift']
X_test.drop('spectrometric_redshift',axis=1,inplace=True)
gbm = lgb.LGBMClassifier(learning_rate = 0.30, metric = 'l2')
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=['auc', 'binary_logloss'])

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

accuracy=accuracy_score(y_pred,y_test)
print("Accuracy :",accuracy)
y_pred_2=[]
red_test_2=[]
red_test=red_lab(red_test)
for i in range(len(red_test)):
    if red_test[i]!=3:
        y_pred_2.append(y_pred[i])
        red_test_2.append(red_test[i])
print("validation : ",spearmanr(red_test_2,y_pred_2)[0])
avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        gbm, X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy(), 
        loss='0-1_loss',
        random_seed=123)

print(avg_bias,avg_var)

"""
                Bias          Variance
cat4 : 0.10577689243027888 0.04527390438247012
cat3 : 0.024806201550387597 0.012093023255813951
cat2 : 0.03656307129798903 0.01487202925045704
cat1 : 0.004464285714285714 0.003115079365079366
"""