import pickle
from itertools import cycle
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split

lnpdata = pd.read_csv('./descriptors_1107.csv', encoding='gbk')

data = lnpdata[
    ['MaxEStateIndex', 'MinEStateIndex', 'MinAbsEStateIndex', 'qed', 'MolWt', 'MaxPartialCharge', 'MinPartialCharge',
     'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_MRLOW', 'BalabanJ', 'BertzCT',
     'HallKierAlpha', 'Ipc', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA2', 'PEOE_VSA3',
     'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4',
     'SMR_VSA5', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA8', 'EState_VSA4', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8',
     'EState_VSA9', 'VSA_EState3', 'VSA_EState4', 'VSA_EState8', 'VSA_EState9', 'NumAliphaticCarbocycles',
     'NumAromaticHeterocycles', 'fr_Al_COO', 'fr_ArN', 'fr_Ar_NH', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH1', 'fr_NH2',
     'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_alkyl_carbamate', 'fr_allylic_oxid', 'fr_bicyclic',
     'fr_hdrzine', 'fr_hdrzone', 'fr_imide', 'fr_ketone', 'fr_methoxy', 'fr_nitro', 'fr_oxime', 'fr_para_hydroxylation',
     'fr_phos_acid', 'fr_priamide', 'fr_sulfonamd', 'fr_thiazole', 'fr_urea', 'inject_way']
].astype(float)
target = lnpdata.iloc[:, -3]

target = np.array(target)
target = target[:, np.newaxis]

data = data.values
stdScale = StandardScaler().fit(data)
data = stdScale.transform(data)

enc = OneHotEncoder()
target = enc.fit_transform(target.reshape(-1, 1)).toarray()
target = [list(oh).index(1) for oh in target]


train_data_all, test_data, train_y_all, test_y = \
        train_test_split(data, target, test_size=0.2, random_state=1, shuffle=True, stratify=target)

with open('models/7579.pkl', 'rb') as f:
    best_model = pickle.load(f)

best_model.fit(train_data_all, train_y_all)

yhat = best_model.predict(test_data)
yhat_prob = best_model.predict_proba(test_data)

accuracy = accuracy_score(yhat, test_y)
print("Test acc：", accuracy)


conf_matrix = confusion_matrix(test_y, yhat)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('Figure/confusion_matrix.svg', format='svg')
plt.show()


y = label_binarize(target, classes=[0, 1, 2])
test_y = label_binarize(test_y, classes=[0, 1, 2])

n_classes = y.shape[1]
n_samples, n_features = data.shape

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_y[:, i], yhat_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(test_y.ravel(), yhat_prob.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot all ROC curves
lw = 2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'blue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver operating characteristic curves', fontsize=14)
plt.legend(loc="lower right")
plt.savefig('Figure/roc_curves.svg', format='svg')
plt.show()


