# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 12:39:36 2021

@author: ABC
"""

import numpy as np
import pandas as pd

df = pd.read_csv('creditcard.csv')

df.shape
df.info()
df.describe()
df.head()

#As ID is not a considered feature so drop it

df.drop('ID',axis=1,inplace=True)
df.head()


#Missing values

df.isnull().sum().sort_values(ascending=False)


#Target 

y = df['default payment next month']

df.drop('default payment next month',axis=1,inplace=True)
X = df
X.info()


#apply PCA

from sklearn.decomposition import PCA
pca = PCA(svd_solver='full')
pc = pca.fit_transform(X)
pca.explained_variance_ratio_

pc.shape
type(pc)
print(pc)

pc_df = pd.DataFrame(pc, columns = ['P0C1', 'P0C2','P0C3','P0C4','P0C5','P0C6','P0C7','P0C8','P096','P0C10','P1C1', 'P1C2','P1C3','P1C4','P1C5','P1C6','P1C7','P1C8','P196','P1C10','P2C1', 'P2C2','P2C3'])
pc_df.head()

df_new = pd.DataFrame({'var': pca.explained_variance_ratio_ , 'pc': ['P0C1', 'P0C2','P0C3','P0C4','P0C5','P0C6','P0C7','P0C8','P096','P0C10','P1C1', 'P1C2','P1C3','P1C4','P1C5','P1C6','P1C7','P1C8','P196','P1C10','P2C1', 'P2C2','P2C3']})

import seaborn as sns
sns.barplot(x='pc',y = 'var', data = df_new, color = 'c')



df1 = pc_df.iloc[:,0:2]
df1.head()
X = df1


from sklearn.model_selection._split import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.25,random_state=100)



from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

models = {
    'LogisticRegression' : LogisticRegression(random_state=42),
    'KNeighborsClassifier' : KNeighborsClassifier(),
    'SVC' : SVC(random_state=42),
    'DecisionTreeClassifier' : DecisionTreeClassifier(max_depth=(1),random_state=(42)),
    'DecisionTreeClassifier Entropy' :DecisionTreeClassifier(criterion='entropy'),
    'SGDClassifier' : SGDClassifier(eta0=0.01),
    'RandomForestClassifier' :RandomForestClassifier(max_features=0.5,n_estimators=500),
    }



from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, accuracy_score

def loss(y_true, y_pred, retu=False):
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    loss = log_loss(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    if retu:
        return pre, rec, f1, loss, acc
    else:
        print('  pre: %.3f\n  rec: %.3f\n  f1: %.3f\n  loss: %.3f\n  acc: %.3f' % (pre, rec, f1, loss, acc))


def train_eval_train(models, X, y):
    for name, model in models.items():
        print(name,':')
        model.fit(X, y)
        loss(y, model.predict(X))
        print('-'*30)
        
train_eval_train(models, X_train, y_train)
