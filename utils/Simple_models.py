import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import CatBoostEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from functools import partial
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def simple_model(X,y,X_test,y_test,categorical_features,metric,cv=5,decision='lr',
                 greater_is_better=True,
                 param_search=False):
    
    X = X.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    
    cat_col = categorical_features
    num_col = list(set(X.columns.tolist())-set(cat_col))

    # пайплайн для категориальных признаков
    cat_pipe = Pipeline([('imputer', SimpleImputer(missing_values='?',
                                                   strategy='most_frequent')),
                         ('ohe', CatBoostEncoder())])
    
    # пайплайн для численных признаков
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                         ('scaler', StandardScaler())])
    # соединяем пайплайны вместе
    transformer = ColumnTransformer(
                               transformers=[('cat', cat_pipe, cat_col),
                                             ('num', num_pipe, num_col)], 
                                             remainder='passthrough') 

    if decision=='lr':
        logreg_clf = Pipeline([('transformer', transformer),
                               ('lr_clf', LogisticRegression(random_state=42,
                                            solver='liblinear'))])
        if param_search:
            params_lr={'lr_clf__penalty' : ['l1', 'l2'],
                       'lr_clf__C' : np.arange(0,1,0.1)}
            model_lr = RandomizedSearchCV(logreg_clf, params_lr, cv=cv).fit(X, y)
            print('Лучшее значение метрики и параметров логистической регрессии:')
            print(model_lr.best_score_, model_lr.best_params_,'\n')
            print('Метрика качества на обучающем множестве логистической регрессии:')
            print(round(metric(y,model_lr.predict_proba(X)[:,1]),3))
            print('Метрика качества на тестовом множестве логистической регрессии:')
            print(round(metric(y_test,model_lr.predict_proba(X_test)[:,1]),3))                

        else:
            model_lr = logreg_clf.fit(X, y)
            print('Метрика качества на обучающем множестве логистической регрессии:')
            print(round(metric(y,model_lr.predict_proba(X)[:,1]),3))
            print('Метрика качества на тестовом множестве логистической регрессии:')
            print(round(metric(y_test,model_lr.predict_proba(X_test)[:,1]),3))

#        pickle.dump(model_lr, open('lr.pkl', "wb"))


    if decision=='tree':

        tree_clf = Pipeline([('transformer', transformer),
                             ('dtree_clf', DecisionTreeClassifier(random_state=42))])
        if param_search:
            params_dtree={'dtree_clf__max_depth': np.arange(1,5,1),
                         'dtree_clf__min_samples_split': np.arange(1,5,1),
                         'dtree_clf__min_samples_leaf': np.arange(1,5,1)}

            model_dtree = RandomizedSearchCV(tree_clf, params_dtree, cv=cv).fit(X, y)
            print('Лучшее значение метрики и параметров решающего дерева:')
            print(model_dtree.best_score_, model_dtree.best_params_,'\n')
            print('Метрика качества на обучающем множестве решающего дерева:')
            print(round(metric(y,model_dtree.predict_proba(X)[:,1]),3))
            print('Метрика качества на тестовом множестве логистической регрессии:')
            print(round(metric(y_test,model_dtree.predict_proba(X_test)[:,1]),3)) 

        else:

            model_dtree = tree_clf.fit(X, y)
            print('Метрика качества на обучающем множестве решающего дерева:')
            print(round(metric(y,model_dtree.predict_proba(X)[:,1]),3))
            print('Метрика качества на тестовом множестве логистической регрессии:')
            print(round(metric(y_test,model_dtree.predict_proba(X_test)[:,1]),3))                 

#        pickle.dump(model_dtree, open('dtree.pkl', "wb"))
