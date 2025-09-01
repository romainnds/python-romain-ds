# %% [markdown]
# # Projet 1 : Prédiction d'un cancer. RDS

# %% [markdown]
# ### 1 - Import des modules, data. Affichage des infos.

# %% [markdown]
# 

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv('breast_cancer.csv')
df.head()

# %%
df.shape

# %%
df.info()

# %%
# Je crée une fonction pour savoir les valeurs manquantes.
def valeurs_manquantes(df):
    nbr_abs = df.isnull().sum()
    pct_abs = (nbr_abs/len(df)*100).round(2)
    res = pd.DataFrame({'Variable :': nbr_abs.index,
                        'Nombre de valeurs manquantes':nbr_abs.values,
                        '% de valeurs manquantes':pct_abs.values})
    return res

# %%
# J'affiche les valeurs manquantes.
valeurs_manquantes(df)

# %%
df.describe()

# %% [markdown]
# ### 2 - Preprocessing.

# %%
df.drop('Unnamed: 32', axis=1, inplace=True)

# %%
df.diagnosis.value_counts() # M = Malin (Tumeur cancéreuse), B = Benin (Tumeur pas cancéreuse).

# %%
import warnings
warnings.filterwarnings('ignore')

plt.figure(figsize=(10,8))
sns.countplot(data=df, x='diagnosis', palette='Set2')
plt.title('Répartition des tumeurs bénignes et malignes')
plt.show()

# %%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# %%
df.diagnosis = le.fit_transform(df.diagnosis)
df.diagnosis.value_counts() # 1 = Malin, 0 = Benin

# %%
correlation = df.corr()
plt.figure(figsize=(20,15))
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm')

# %%
df.drop('id', axis=1, inplace=True)

# %%
# Selection des X.
correlation = df.corr().abs()

# Tri
upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
todrop = [column for column in upper.columns if any(upper[column]>0.92)]

new_df = df.drop(columns=todrop).copy()

# %%
new_df.head()

# %%
new_df.shape

# %%
X = new_df.drop('diagnosis', axis=1)
y = new_df.diagnosis

# %%
# Je split mon modèle
from sklearn.model_selection import train_test_split as tts 
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, stratify=y)


# %%
# Moyenne 0 ecart-type 1 de mes features.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# %%
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.fit_transform(X_test)


# %% [markdown]
# ## Entraînement du modèle.

# %%
# SVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svc = SVC(probability=True)

param_grid = {
    'C':[0.01 ,0.05 ,0.1 ,0.5 ,1, 5],
    'kernel':['linear', 'poly', 'rbf'],
    'gamma':['scale', 'auto', 'poly']
}

grid_searchcv = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1, scoring='accuracy')

grid_searchcv.fit(X_train_s, y_train)

# %%
grid_searchcv.best_params_

# %%
grid_searchcv.best_score_

# %%
best_model = grid_searchcv.best_estimator_
test_score = best_model.score(X_test_s, y_test)


# %%
test_score

# %%


# %%



