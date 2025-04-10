import shap
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle


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

feature_names = data.columns

# data = data.values
stdScale = StandardScaler().fit(data)
data = stdScale.transform(data)

data = np.round(data, 3)
data = pd.DataFrame(data, columns=feature_names)

target = np.array(target)
target = target[:, np.newaxis]
enc = OneHotEncoder()
target = enc.fit_transform(target.reshape(-1, 1)).toarray()
target = [list(oh).index(1) for oh in target]

train_data_all, test_data, train_y_all, test_y = \
    train_test_split(data, target, test_size=0.2, random_state=1024, shuffle=True, stratify=target)

with open('models/7579.pkl', 'rb') as f:
    model = pickle.load(f)


def model_wrapper(X):
    return model.predict_proba(X)


explainer = shap.Explainer(model_wrapper, train_data_all)

shap_values = explainer(test_data)

print(test_data)
print(test_y)

# =================================================================
# importance plot

shap.summary_plot(shap_values[:, :, 0], feature_names=feature_names, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig('./Figure/importance_0.svg', dpi=600, format='svg', transparent=False, bbox_inches='tight')
plt.clf()

shap.summary_plot(shap_values[:, :, 1], feature_names=feature_names, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig('./Figure/importance_1.svg', dpi=600, format='svg', transparent=False, bbox_inches='tight')
plt.clf()

shap.summary_plot(shap_values[:, :, 2], feature_names=feature_names, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig('./Figure/importance_2.svg', dpi=600, format='svg', transparent=False, bbox_inches='tight')
plt.clf()

# =================================================================
# force plot

shap.plots.force(shap_values[0, :, 0], feature_names=feature_names, show=False, matplotlib=True, figsize=(20, 5))
ax = plt.gca()

for text in ax.texts:
    text.set_fontsize(16)

plt.tight_layout()
plt.savefig('./Figure/force_0_0.svg', dpi=600, format='svg', transparent=False, bbox_inches='tight')
plt.clf()

shap.plots.force(shap_values[0, :, 1], feature_names=feature_names, show=False, matplotlib=True, figsize=(20, 5))
ax = plt.gca()

for text in ax.texts:
    text.set_fontsize(15)
plt.tight_layout()
plt.savefig('./Figure/force_0_1.svg', dpi=600, format='svg', transparent=False, bbox_inches='tight')
plt.clf()

shap.plots.force(shap_values[0, :, 2], feature_names=feature_names, show=False, matplotlib=True, figsize=(20, 5))
ax = plt.gca()


for text in ax.texts:
    text.set_fontsize(16)
plt.tight_layout()
plt.savefig('./Figure/force_0_2.svg', dpi=600, format='svg', transparent=False, bbox_inches='tight')
plt.clf()

# =================================================================
# summary plot

shap.summary_plot(shap_values[:, :, 0], feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig('./Figure/summary_0.svg', dpi=600, format='svg', transparent=False, bbox_inches='tight')
plt.clf()

shap.summary_plot(shap_values[:, :, 1], feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig('./Figure/summary_1.svg', dpi=600, format='svg', transparent=False, bbox_inches='tight')
plt.clf()

shap.summary_plot(shap_values[:, :, 2], feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig('./Figure/summary_2.svg', dpi=600, format='svg', transparent=False, bbox_inches='tight')
plt.clf()



