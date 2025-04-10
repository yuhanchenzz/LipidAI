"""
# This source code is modified from https://github.com/aspuru-guzik-group/long-acting-injectables
# under MIT License. The original license is included below:
# ========================================================================

MIT License

Copyright (c) 2022 Aspuru-Guzik group repo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV


def NESTED_CV(X, Y, model_type, n_iter):
    if model_type == 'LightGBM':
        user_defined_model = LGBMClassifier(objective='multiclass')
        param_grid = {'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1],
                      'n_estimators': [20, 50, 100, 200, 500],
                      'boosting_type': ['gbdt', 'dart', 'goss'],
                      'num_leaves': [16, 32, 64, 128, 256],
                      'reg_alpha': [0, 0.005, 0.01, 0.015],
                      'reg_lambda': [0, 0.005, 0.01, 0.015],
                      'min_child_weight': [0.001, 0.01, 0.1, 1.0, 10.0],
                      'subsample': [0.4, 0.6, 0.8, 1.0],
                      'min_child_samples': [2, 10, 20, 40, 100]
                      }

    elif model_type == 'DecisionTree':
        user_defined_model = DecisionTreeClassifier(random_state=1024)
        param_grid = {'criterion': ['gini', 'entropy'],
                      'splitter': ['best', 'random'],
                      'max_depth': [None],
                      'min_samples_split': [2, 4, 6],
                      'min_samples_leaf': [1, 2, 4],
                      'max_features': [None, 'auto', 'sqrt', 'log2'],
                      'ccp_alpha': [0, 0.05, 0.1, 0.15]}

    elif model_type == 'KNN':
        user_defined_model = KNeighborsClassifier()
        param_grid = {'n_neighbors': [2, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 50],
                      'weights': ["uniform", 'distance'],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                      'leaf_size': [10, 30, 50, 75, 100],
                      'p': [1, 2],
                      'metric': ['minkowski']}

    elif model_type == 'LogisticRegression':
        user_defined_model = LogisticRegression(random_state=1024, fit_intercept=True, max_iter=500)
        param_grid = [
            {'solver': ['newton-cg', 'lbfgs', 'sag'], 'penalty': ['l2'], 'C': np.arange(0.05, 2, 0.05).tolist(),
             'multi_class': ['auto', 'ovr', 'multinomial'], 'class_weight': [None, 'balanced']},

            {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'C': np.arange(0.05, 2, 0.05).tolist(),
             'multi_class': ['ovr'], 'class_weight': [None, 'balanced']},

            {'solver': ['saga'], 'penalty': ['l1', 'l2'], 'C': np.arange(0.05, 2, 0.05).tolist(),
             'multi_class': ['auto', 'ovr', 'multinomial'], 'class_weight': [None, 'balanced']},

            {'solver': ['saga'], 'penalty': ['elasticnet'], 'C': np.arange(0.05, 2, 0.05).tolist(),
             'multi_class': ['auto', 'ovr', 'multinomial'], 'class_weight': [None, 'balanced'],
             'l1_ratio': np.arange(0.1, 1, 0.1).tolist()}
        ]

    elif model_type == 'RF':
        user_defined_model = RandomForestClassifier(random_state=1024)
        param_grid = {'n_estimators': [100, 300, 400],
                      'criterion': ['gini', 'entropy'],
                      'max_depth': [None],
                      'min_samples_split': [2, 4, 6, 8],
                      'min_samples_leaf': [1, 2, 4],
                      'min_weight_fraction_leaf': [0.0],
                      'max_features': ['auto', 'sqrt'],
                      'max_leaf_nodes': [None],
                      'min_impurity_decrease': [0.0],
                      'bootstrap': [True],
                      'oob_score': [True],
                      'ccp_alpha': [0, 0.005, 0.01]}

    elif model_type == 'SVM':
        user_defined_model = SVC(random_state=1024)
        param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                      'degree': [2, 3, 4, 5, 6],
                      'gamma': ['scale', 'auto'],
                      'C': [0.1, 0.5, 1, 2],
                      'shrinking': [True, False]}

    elif model_type == 'XGBoost':
        user_defined_model = XGBClassifier(use_label_encoder=False)
        param_grid = {'booster': ['gbtree', 'gblinear', 'dart'],
                      "n_estimators": [100, 150, 300, 400],
                      'max_depth': [6, 7, 8, 9],
                      'gamma': [0, 2, 4, 6, 8, 10],  # min_split_loss
                      'min_child_weight': [1.0, 2.0, 4.0, 5.0],
                      'max_delta_step': [1, 2, 4, 6, 8, 10],
                      'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                      'learning_rate': [0.3, 0.2, 0.1, 0.05, 0.01],
                      'reg_alpha': [0.001, 0.01, 0.1],
                      'reg_lambda': [0.001, 0.01, 0.1],
                      'scale_pos_weight': [1, 2, 3]}

    elif model_type == 'AdaBoost':
        user_defined_model = AdaBoostClassifier(random_state=1024)
        param_grid = {'n_estimators': [50, 100, 200],
                      'learning_rate': [1.0, 0.5, 0.1],
                      'algorithm': ['SAMME', 'SAMME.R'],
                      'base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)],
                      }

    elif model_type == 'GradientBoost':
        user_defined_model = GradientBoostingClassifier(random_state=1024)
        param_grid = {'n_estimators': [200, 400, 600],
                      'learning_rate': [0.01, 0.05, 0.1],
                      'max_depth': [3, 4, 5],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 2, 4],
                      'subsample': [0.8, 0.9, 1.0],
                      'max_features': ['auto', 'sqrt', 'log2', None]}

    elif model_type == 'GaussianNB':
        user_defined_model = GaussianNB()

    else:
        print("model type error!")

    iter_number = []
    outer_results = []
    inner_results = []
    model_param = []
    test_y_list = []
    pred_list = []

    for i in range(n_iter):
        # outer-loop
        train_data_all, test_data, train_y_all, test_y = \
            train_test_split(X, Y, test_size=0.2, random_state=i, shuffle=True, stratify=Y)

        test_y_list.append(np.array(test_y))

        # inner-loop
        cv_inner = StratifiedKFold(n_splits=10)

        search = RandomizedSearchCV(user_defined_model, param_grid, n_iter=100, verbose=0, cv=cv_inner, refit=True,
                                    scoring='accuracy', n_jobs=-1)

        result = search.fit(train_data_all, train_y_all)

        best_model = result.best_estimator_

        inner_results.append(result.best_score_)

        yhat = best_model.predict(test_data)

        pred_list.append(yhat)

        acc = accuracy_score(yhat, test_y)

        iter_number.append(i + 1)
        outer_results.append(acc)
        model_param.append(result.best_params_)

        print('\n################################################################\n\nSTATUS REPORT:')
        print('Iteration ' + str(i + 1) + ' of ' + str(n_iter) + ' runs completed')
        print('Test_Score: %.3f, Best_Valid_Score: %.3f, \n\nBest_Model_Params: \n%s' % (
            acc, result.best_score_, result.best_params_))
        print("\n################################################################\n ")

    list_of_tuples = list(zip(iter_number, inner_results, outer_results, model_param, test_y_list, pred_list))
    CV_dataset = pd.DataFrame(list_of_tuples,
                              columns=['Iter', 'Valid Score', 'Test score', 'Model Params', 'True classify',
                                       'Predicted classify'])
    CV_dataset.to_excel("./base_model_training/model_contrast_extra_" + str(model_type) + ".xlsx", index=False)


lnpdata = pd.read_csv('./descriptors_1107.csv', encoding='gbk')

data = lnpdata.iloc[:, :-3].astype(float)
target = lnpdata.iloc[:, -3]

target = np.array(target)
target = target[:, np.newaxis]

data = data.values
stdScale = StandardScaler().fit(data)
data = stdScale.transform(data)

enc = OneHotEncoder()
target = enc.fit_transform(target.reshape(-1, 1)).toarray()
target = [list(oh).index(1) for oh in target]

NESTED_CV(data, target, 'LightGBM', 20)
