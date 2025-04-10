import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

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

with open('model/7579.pkl', 'rb') as f:
    model = pickle.load(f)


cv = StratifiedKFold(n_splits=10, random_state=1024, shuffle=True)
# accuracy, precision, recall, f1
acc = cross_val_score(model, data, target, cv=cv, scoring='accuracy', n_jobs=-1)
precision = cross_val_score(model, data, target, cv=cv, scoring='precision_macro', n_jobs=-1)
recall = cross_val_score(model, data, target, cv=cv, scoring='recall_macro', n_jobs=-1)
f1 = cross_val_score(model, data, target, cv=cv, scoring='f1_macro', n_jobs=-1)

print("10-fold acc：", np.mean(acc))
print("10-fold precision：", np.mean(precision))
print("10-fold recall：", np.mean(recall))
print("10-fold f1_score：", np.mean(f1))

