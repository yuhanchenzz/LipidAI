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

import ast
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


lnpdata = pd.read_csv('./descriptors_1107.csv', encoding='gbk')

data = lnpdata.iloc[:, :-3].astype(float)
data_features = data
data = data.values
stdScale = StandardScaler().fit(data)
data = stdScale.transform(data)

target = lnpdata.iloc[:, -3]
target = np.array(target)
target = target[:, np.newaxis]

enc = OneHotEncoder()
target = enc.fit_transform(target.reshape(-1, 1)).toarray()
target = [list(oh).index(1) for oh in target]

target = np.array(target)
# ========================================================================
# optional models

# LightGBM
lightgbm_data = pd.read_excel('./base_model_training/model_contrast_LightGBM.xlsx')
lightgbm_data['score_difference'] = abs(lightgbm_data['Valid Score'] - lightgbm_data['Test score'])
lightgbm_data = lightgbm_data.sort_values(by=['Test score', 'Valid Score'], ascending=False)
# lightgbm_data = lightgbm_data.sort_values(by=['Valid Score', 'Test score'], ascending=False)
# lightgbm_data = lightgbm_data.sort_values(by=['score_difference'], ascending=True)
best_param_lgb = lightgbm_data.iloc[0, 3]
best_param_lgb = ast.literal_eval(best_param_lgb)
model_lgb = LGBMClassifier(objective='multiclass', **best_param_lgb)

# RF
rf_data = pd.read_excel('./base_model_training/model_contrast_RF.xlsx')
rf_data['score_difference'] = abs(rf_data['Valid Score'] - rf_data['Test score'])
# rf_data = rf_data.sort_values(by=['Test score', 'Valid Score'], ascending=False)
rf_data = rf_data.sort_values(by=['Valid Score', 'Test score'], ascending=False)
# rf_data = rf_data.sort_values(by=['score_difference'], ascending=True)
best_param_rf = rf_data.iloc[0, 3]
best_param_rf = ast.literal_eval(best_param_rf)
model_rf = RandomForestClassifier(**best_param_rf, random_state=1024)

# XGBoost
xgb_data = pd.read_excel('./base_model_training/model_contrast_XGBoost.xlsx')
xgb_data['score_difference'] = abs(xgb_data['Valid Score'] - xgb_data['Test score'])
# xgb_data = xgb_data.sort_values(by=['Test score', 'Valid Score'], ascending=False)
# xgb_data = xgb_data.sort_values(by=['Valid Score', 'Test score'], ascending=False)
xgb_data = xgb_data.sort_values(by=['score_difference'], ascending=True)
best_param_xgb = xgb_data.iloc[0, 3]
best_param_xgb = ast.literal_eval(best_param_xgb)
model_xgb = XGBClassifier(**best_param_xgb, use_label_encoder=False)


# GBDT
gbdt_data = pd.read_excel('./base_model_training/model_contrast_GradientBoost.xlsx')
gbdt_data['score_difference'] = abs(gbdt_data['Valid Score'] - gbdt_data['Test score'])
# gbdt_data = gbdt_data.sort_values(by=['Test score', 'Valid Score'], ascending=False)
# gbdt_data = gbdt_data.sort_values(by=['Valid Score', 'Test score'], ascending=False)
gbdt_data = gbdt_data.sort_values(by=['score_difference'], ascending=True)
best_param_gbdt = gbdt_data.iloc[0, 3]
best_param_gbdt = ast.literal_eval(best_param_gbdt)
model_gbdt = GradientBoostingClassifier(**best_param_gbdt, random_state=1024)

# DecisionTree
dt_data = pd.read_excel('./base_model_training/model_contrast_DecisionTree.xlsx')
dt_data['score_difference'] = abs(dt_data['Valid Score'] - dt_data['Test score'])
dt_data = dt_data.sort_values(by=['Test score', 'Valid Score'], ascending=False)
# dt_data = dt_data.sort_values(by=['Valid Score', 'Test score'], ascending=False)
# dt_data = dt_data.sort_values(by=['score_difference'], ascending=True)
best_param_dt = dt_data.iloc[0, 3]
best_param_dt = ast.literal_eval(best_param_dt)
model_dt = DecisionTreeClassifier(**best_param_dt, random_state=1024)

# KNN
knn_data = pd.read_excel('./base_model_training/model_contrast_KNN.xlsx')
knn_data['score_difference'] = abs(knn_data['Valid Score'] - knn_data['Test score'])
# knn_data = knn_data.sort_values(by=['Test score', 'Valid Score'], ascending=False)
# knn_data = knn_data.sort_values(by=['Valid Score', 'Test score'], ascending=False)
knn_data = knn_data.sort_values(by=['score_difference'], ascending=True)
best_param_knn = knn_data.iloc[0, 3]
best_param_knn = ast.literal_eval(best_param_knn)
model_knn = KNeighborsClassifier(**best_param_knn)

# LogisticRegression
lr_data = pd.read_excel('./base_model_training/model_contrast_LogisticRegression')
lr_data['score_difference'] = abs(lr_data['Valid Score'] - lr_data['Test score'])
lr_data = lr_data.sort_values(by=['Test score', 'Valid Score'], ascending=False)
# lr_data = lr_data.sort_values(by=['Valid Score', 'Test score'], ascending=False)
# lr_data = lr_data.sort_values(by=['score_difference'], ascending=True)
best_param_lr = lr_data.iloc[0, 3]
best_param_lr = ast.literal_eval(best_param_lr)
model_lr = LogisticRegression(random_state=1024, fit_intercept=True, max_iter=500, **best_param_lr)

# SVM
svm_data = pd.read_excel('./base_model_training/model_contrast_SVM.xlsx')
svm_data['score_difference'] = abs(svm_data['Valid Score'] - svm_data['Test score'])
# svm_data = svm_data.sort_values(by=['Test score', 'Valid Score'], ascending=False)
# svm_data = svm_data.sort_values(by=['Valid Score', 'Test score'], ascending=False)
svm_data = svm_data.sort_values(by=['score_difference'], ascending=True)
best_param_svm = svm_data.iloc[0, 3]
best_param_svm = ast.literal_eval(best_param_svm)
model_svm = SVC(**best_param_svm, probability=True, random_state=1024)

# ===========================================================================
# default models

model_lgb_2 = LGBMClassifier(objective='multiclass')
model_dt_2 = DecisionTreeClassifier(random_state=4)
model_knn_2 = KNeighborsClassifier()
model_rf_2 = RandomForestClassifier(random_state=4)
model_svm_2 = SVC(probability=True, random_state=4)
model_xgb_2 = XGBClassifier(use_label_encoder=False)
model_gbdt_2 = GradientBoostingClassifier(random_state=4)
model_lr_2 = LogisticRegression()

# ===========================================================================
# stacking

lr = LogisticRegression(random_state=4, fit_intercept=True, max_iter=500, solver='saga', penalty='elasticnet',
                        multi_class='ovr', l1_ratio=0.8, class_weight='None', C=1.95)

sclf = StackingCVClassifier(classifiers=[model_lgb, model_rf, model_xgb, model_gbdt, model_dt, model_knn,
                                         model_lgb_2, model_xgb_2, model_gbdt_2, model_rf_2,
                                         ],
                            use_probas=True,
                            meta_classifier=lr,
                            random_state=1024)
# ===========================================================================
# model refinement

# Ward_linkage
corr = spearmanr(data).correlation
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)
distance_matrix = 1 - np.abs(corr)

dist_linkage = hierarchy.ward(squareform(distance_matrix))

ACC_list = []
std_list = []
test_test_list = []
feature_name_list = []
feature_number_list = []
linkage_distance_list = []

for n in range(0, 174, 1):
    cluster_ids = hierarchy.fcluster(dist_linkage, (n / 174), criterion="distance")

    cluster_id_to_feature_ids = defaultdict(list)

    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)

    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    linkage_distance_list.append(n / 174)  # append linkage distance to empty list
    tested_features = []

    for feature in selected_features:  # for loop to append the utilized input feature names to the empty list
        tested_features.append(data_features.columns[feature])

    feature_number_list.append(len(tested_features))  # append the number of input features to empty list
    feature_name_list.append(tested_features)  # append the list of feature names to an empty list of lists

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1024)
    scores = cross_val_score(sclf, data[:, selected_features], target, cv=cv, scoring='accuracy', n_jobs=-1)

    ACC_list.append(np.mean(scores))  # append average MAE value to empty list
    std_list.append(np.std(scores))
    test_test_list.append(scores)

    print('\n################################################################\n\nSTATUS REPORT:')
    print('Iteration ' + str(n + 1) + ' of ' + str(174) + ' completed')
    print('Test_Score: %.3f' % (np.mean(scores)))
    print("\n################################################################\n ")

# results dataframe
list_of_tuples = list(zip(feature_number_list, feature_name_list, ACC_list, std_list, test_test_list,
                          linkage_distance_list))  # create a list of tuples with results model refinement
results_df = pd.DataFrame(list_of_tuples, columns=['# of Features', 'Feature names', 'ACC', 'std', 'test_values',
                                                   'linkage distance'])

results_df.to_excel("Model_Refinement_1.xlsx")

