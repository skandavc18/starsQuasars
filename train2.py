import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('cat4.csv')


data.drop('Unnamed: 0', axis = 1, inplace = True)
data.drop('spectrometric_redshift', axis = 1, inplace = True) #drop redshift else problem is trivial

print(data.head())


X = data.drop(['pred'], axis=1)
y = data['pred']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=20)


gbm = lgb.LGBMClassifier(learning_rate = 0.22, metric = 'l1')
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=['auc', 'binary_logloss'],
early_stopping_rounds=5)

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

accuracy=accuracy_score(y_pred,y_test)
print("Accuracy :",accuracy)