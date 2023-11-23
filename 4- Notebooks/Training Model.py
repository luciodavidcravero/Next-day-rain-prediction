#!/usr/bin/env python
# coding: utf-8

# ## Rain prediction in Australia

# #### Import required libraries

# In[40]:


import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from statsmodels.stats.outliers_influence import variance_inflation_factor
from imblearn.over_sampling import SMOTE

import sklearn

from sklearn.svm import SVC
from sklearn.svm import SVR

from sklearn.impute import SimpleImputer

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler,OneHotEncoder, LabelEncoder

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score, recall_score, f1_score,ConfusionMatrixDisplay,classification_report
from sklearn.metrics import mean_squared_error

import optuna

import xgboost as xgb
from xgboost import XGBClassifier

import joblib


# <br>
# <br>
# <br>
# <br>
# <br>

# #### Read dataset

# In[50]:


dataframe_clean_wo_outl_wo_corr = pd.read_csv(r"C:\Users\Lucio\Documents\Github\Next-day-rain-prediction\1- Data\2- Processed\dataframe_clean_wo_outl_wo_corr.csv", index_col=0)
dataframe_clean_wo_outl_wo_corr.head()


# dataframe_clean_wo_outl_wo_corr characteristics:
# - Removed univariated outliers
# - Removed variables with high collinearity

# <br>
# <br>
# <br>
# <br>
# <br>

# #### Encode Categorical Features

# In[51]:


dataframe_encoded = pd.get_dummies(dataframe_clean_wo_outl_wo_corr)
dataframe_encoded.head()


# <br>
# <br>
# <br>
# <br>
# <br>

# ## Model Tranining

# #### Create X and y dataframes

# In[52]:


X = dataframe_encoded[[c for c in dataframe_encoded if c != 'RainTomorrow']].values
y = dataframe_encoded[['RainTomorrow']]


# In[53]:


X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y,random_state=42, test_size=0.30)


# #### Define optimization function

# In[54]:


def objective_xgb(trial):
    param = {
        "verbosity": 1,
        "objective": "binary:logistic",
        "booster": trial.suggest_categorical("booster", ["gbtree"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        'eval_metric': 'error',
    }

    #param["booster"] == "gbtree"
    param["subsample"] = trial.suggest_float("subsample", 1e-8, 1.0, log=True)
    param["n_estimators"] = trial.suggest_int("n_estimators", 1, 1000)        
    param["max_depth"] = trial.suggest_int("max_depth", 1, 64)
    param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
    param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
    param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    
    ratio_majority_to_minority = len(y_train[y_train == 0]) / len(y_train[y_train == 1])  #Adjust weights based on rain/no-rain proportion
    
    bst = xgb.XGBClassifier(**param, scale_pos_weight=ratio_majority_to_minority)
    bst.fit(X_train, y_train)

    y_pred = bst.predict(X_val)
    accuracy = sklearn.metrics.accuracy_score(y_val, y_pred)

    return -accuracy  #Negative accuracy to maximize it (because 'eval_metric': 'error')


# #### Applying StandardScaler

# In[55]:


sc_X = MinMaxScaler()
sc_y = MinMaxScaler()
X_sc = sc_X.fit_transform(X)
y_sc = sc_y.fit_transform(y)


# In[56]:


X_sc_train, X_sc_val, y_sc_train, y_sc_val = sklearn.model_selection.train_test_split(X_sc, y_sc, random_state=42, test_size=0.30)


# #### Applying SMOTE

# In[57]:


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_sc_train, y_train)


# #### Hyperparameter optimization with Optuna

# In[58]:


study_xgb = optuna.create_study()
study_xgb.optimize(objective_xgb, n_trials=10)
study_xgb.best_params


# In[59]:


xgb_params = study_xgb.best_params
xgb_params


# #### Train model using best parameters
# Adjust weights based on rain/no-rain proportion

# In[60]:


ratio_majority_to_minority = len(y_sc_train[y_sc_train == 0]) / len(y_sc_train[y_sc_train == 1])  #Adjust weights based on rain/no-rain proportion

model = XGBClassifier(**xgb_params, silent=0, scale_pos_weight=ratio_majority_to_minority)

#Sin SMOTE
#model.fit(X_sc_train, y_sc_train, eval_set=[(X_sc_val, y_sc_val)], early_stopping_rounds=10, verbose=True)

#Con SMOTE
model.fit(X_train_resampled, y_train_resampled, eval_set=[(X_sc_val, y_val)], early_stopping_rounds=10, verbose=True)


# #### Predict using validation dataset

# In[ ]:


y_predicted = model.predict(X_sc_val)
y_predicted


# #### Model performance evaluation

# In[ ]:


conf_matrix = confusion_matrix(y_val, y_predicted)

accuracy = accuracy_score(y_val, y_predicted)
precision = precision_score(y_val, y_predicted)
recall = recall_score(y_val, y_predicted)
f1 = f1_score(y_val, y_predicted)
roc_auc = roc_auc_score(y_val, y_predicted)

print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("ROC AUC:", roc_auc)


# #### Save model

# In[ ]:


ubi = r'C:\Users\Lucio\Documents\Github\Next-day-rain-prediction\3- Models/XGBClf_rain_pred.joblib'

joblib.dump(model, ubi)


# Sin weights:
#     Confusion Matrix:
#  [[32302  1817]
#  [ 4731  4749]]
# Accuracy: 0.8498130691070896
# Precision: 0.7232713981114834
# Recall: 0.5009493670886076
# F1-Score: 0.5919232207403714
# ROC AUC: 0.7238472911822768
# 
# Con weights:
# Confusion Matrix:
#  [[27030  7089]
#  [ 2289  7191]]
# Accuracy: 0.7849033234707218
# Precision: 0.5035714285714286
# Recall: 0.7585443037974684
# F1-Score: 0.6053030303030303
# ROC AUC: 0.7753857542903635
