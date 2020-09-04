#!/usr/bin/env python
# coding: utf-8

# Objective: Predict prices for test (test.csv) dataset based on model build from train (train.csv) dataset  
# 
# Evaluation metric: "The RMSE between log of Saleprice and log of prediction". Need to convert salesprice to log value first. However seems that BoxCox does a better job here. For my testing E will remove BoxCox, but may want to put it back for the final submissions. Maybe one with boxcox1p() and one with log().
# 
# Original competition (explains the evaluation metric): https://www.kaggle.com/c/home-data-for-ml-course/overview/evaluation. My work is paying off, my submission on that site is #3 out of 38000, top .0001%
# 
# Original notebook source: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard  
# 
# Reference for good ideas: https://towardsdatascience.com/tricks-i-used-to-succeed-on-a-famous-kaggle-competition-adaa16c09f22
# 
# Original default score was .11543, new best score is 0.11353
# 
# Stacked and Ensembled Regressions to predict House Prices  
# How to Kaggle: https://www.youtube.com/watch?v=GJBOMWpLpTQ
# 
# References for stacking and ensembling:
# https://www.kaggle.com/getting-started/18153
# https://developer.ibm.com/technologies/artificial-intelligence/articles/stack-machine-learning-models-get-better-results
# 
# Donald S  
# July 2020  
# 
# Need to submit every 2 months as the leaderboard will rollover after this 2 month period

# <pre>
# Typical flow of model building: Use GridSearch for determining Best parameters => **best_estimator_ 
# 
# good basic resource for models: 
# https://scikit-learn.org/stable/modules/cross_validation.html
# https://scikit-learn.org/stable/modules/grid_search.html
# 
# A search consists of:
# an estimator (regressor or classifier such as sklearn.svm.SVC());  
# a parameter space;  
# a method for searching or sampling candidates;  
# a cross-validation scheme; and  
# a score function.  ****
# 

# ![image.png](attachment:image.png)

# We will be using Ensembling methods in this notebook. I will only create one dataset for all models to use, but you can create multiple datasets, each to be used by a different set of models, depending on your use case.
# 
# ![image.png](attachment:image.png)

# In[1]:


#import some necessary libraries

import numpy as np # linear algebra
numpy_seed = 0
np.random.seed(numpy_seed) # numpy
import random as rn
random_seed = 0
rn.seed(random_seed)

import tensorflow as tf
tf_seed = 0
tf.random.set_seed(tf_seed) # tf

import datetime
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
#import warnings
#def ignore_warn(*args, **kwargs):
#    pass
#warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
import time
import math

from scipy import stats
from scipy.stats import norm, skew, kurtosis, boxcox #for some statistics
from scipy.special import boxcox1p, inv_boxcox, inv_boxcox1p
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
lam_l = 0.35
competition = 'SR'

from subprocess import check_output
#print(check_output(["ls", "-rlt", "../StackedRegression"]).decode("utf8")) #check the files available in the directory
#print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory

StartTime = datetime.datetime.now()


# In[2]:


class MyTimer():
    # usage:
    #with MyTimer():                            
    #    rf.fit(X_train, y_train)
    
    def __init__(self):
        self.start = time.time()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        runtime = end - self.start
        msg = 'The function took {time} seconds to complete'
        print(msg.format(time=runtime))


# **Import libraries**

# In[3]:

from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

#from sklearn.metrics import mean_squared_log_error
# to run locally: conda install -c anaconda py-xgboost
import xgboost as xgb
import lightgbm as lgb
from catboost import Pool, CatBoostRegressor


# In[4]:


train = pd.read_csv('HousePricesTrain.csv')
test = pd.read_csv('HousePricesTest.csv')
y_train = boxcox1p(train['SalePrice'], lam_l) 

train.drop("SalePrice", axis = 1, inplace = True)

train_ID = train['Id']
test_ID = test['Id']

train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)


# In[5]:


subtrain, val, y_subtrain, y_val = train_test_split(train, y_train, test_size=0.33, random_state=42)


# In[6]:


train.head()


# In[7]:


test.head()


# **Define a cross validation strategy**

# We use the **cross_val_score** function of Sklearn. However this function has no shuffle attribute, so we add one line of code,  in order to shuffle the dataset  prior to cross-validation

# replace cross_val_score() with cross_validate()
# # reference: https://scikit-learn.org/stable/modules/cross_validation.html
# 
#     from sklearn.metrics import make_scorer
#     scoring = {'prec_macro': 'precision_macro',
#                'rec_macro': make_scorer(recall_score, average='macro')}
#     scores = cross_validate(clf, X, y, scoring=scoring,
#                             cv=5, return_train_score=True)
#     sorted(scores.keys())
#     ['fit_time', 'score_time', 'test_prec_macro', 'test_rec_macro',
#      'train_prec_macro', 'train_rec_macro']
#     scores['train_rec_macro']
#     array([0.97..., 0.97..., 0.99..., 0.98..., 0.98...])

# Kfold is useful for thorough testing of a model, will give a more accurate score based on remove some data test on the remaining and change the data removed each time. See image below for details:

# ![image.png](attachment:image.png)

# In[8]:


#Validation function
# train.values and y_train are both log scaled so just need to take the square of the delta between them to calculate the error, then take the sqrt to get rmsle
# but for now y_train is boxcox1p(), not log(). Use this to convert back: inv_boxcox1p(y_train, lam_l)
n_folds=5 # only for more accurate comparision, the number here does not affect the final score

def rmsle_cv(model):
    print("running rmsle_cv code")
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values) # was 42
    # other scores: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf)) # also r2
    print("raw rmse scores for each fold:", rmse)
    return(rmse)

def r2_cv(model):
    print("running r2_cv code")
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values) # was 42
    # other scores: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    r2= cross_val_score(model, train.values, y_train, scoring="r2", cv = kf) # also r2
    print("raw r2 scores for each fold:", r2)
    return(r2)

# used for another competition
def mae_cv(model):
    print("running mae_cv code")
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values) # was 42
    # other scores: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    mae = -cross_val_score(model, train.values, y_train, scoring="neg_mean_absolute_error", cv = kf) # also r2
    print("raw mae scores for each fold:", mae)
    return(mae)

def all_cv(model, n_folds, cv):
    print("running cross_validate")
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values) # was 42
    # other scores: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    scorers = {
        'r2': 'r2',
        'nmsle': 'neg_mean_squared_log_error', 
        'nmse': 'neg_mean_squared_error',
        'mae': 'neg_mean_absolute_error'
    }
    scores = cross_validate(model, train.values, y_train, scoring=scorers,
                           cv=kf, return_train_score=True)
    return(scores)


# ##Base models

# In[9]:


def runGSCV(num_trials, features, y_values):
    non_nested_scores = np.zeros(num_trials) # INCREASES BIAS
    nested_scores = np.zeros(num_trials)
    # Loop for each trial
    for i in range(num_trials):
        print("Running GridSearchCV:")
        with MyTimer():    
            #grid_result = gsc.fit(train, y_train)  
            grid_result = gsc.fit(features, y_values)  
        non_nested_scores[i] = grid_result.best_score_
        if (competition == 'SR'):
            print("Best mae %f using %s" % ( -grid_result.best_score_, grid_result.best_params_))
        else:
            print("Best rmse %f using %s" % ( np.sqrt(-grid_result.best_score_), grid_result.best_params_))
        
        # nested/non-nested cross validation: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
        with MyTimer():    
            #nested_score = cross_val_score(gsc, X=train, y=y_train, cv=outer_cv, verbose=0).mean() 
            nested_score = cross_val_score(gsc, X=features, y=y_values, cv=outer_cv, verbose=0).mean() 
            # source code for cross_val_score is here: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_validation.py#L137
        if (competition == 'SR'):
            print("nested mae score from KFold %0.3f" % -nested_score)
        else:
            print("nested rmse score from KFold %0.3f" % np.sqrt(-nested_score))
        
        nested_scores[i] = nested_score
        print('grid_result',grid_result)
        #if (competition == 'SR'):
        print("mean scores: r2(%0.3f) mae(%0.3f) nmse(%0.3f) nmsle(%0.3f)" % (grid_result.cv_results_['mean_test_r2'].mean(), -grid_result.cv_results_['mean_test_mae'].mean(),  np.sqrt(-grid_result.cv_results_['mean_test_nmse'].mean()), grid_result.cv_results_['mean_test_nmsle'].mean() ))
        #print("mean scores: r2(%0.3f) nmse(%0.3f) mae(%0.3f)" % (grid_result.cv_results_['mean_test_r2'].mean(), np.sqrt(-grid_result.cv_results_['mean_test_nmse'].mean()), grid_result.cv_results_['mean_test_mae'].mean()))
    return grid_result


# In[10]:


def calc_all_scores(model, n_folds=5, cv=5):
    scores = all_cv(model, n_folds, cv)
    #scores['train_<scorer1_name>'']
    #scores['test_<scorer1_name>'']
    print("\n mae_cv score: {:.4f} ({:.4f})\n".format( (-scores['test_mae']).mean(), scores['test_mae'].std() ))
    print("\n rmsle_cv score: {:.4f} ({:.4f})\n".format( (np.sqrt(-scores['test_nmse'])).mean(), scores['test_nmse'].std() ))
    print("\n r2_cv score: {:.4f} ({:.4f})\n".format( scores['test_r2'].mean(), scores['test_r2'].std() ))
    return (scores)

# useful when you can't decide on parameter setting from best_params_
# result_details(grid_result,'mean_test_nmse',100)
def result_details(grid_result,sorting='mean_test_nmse',cols=100):
    param_df = pd.DataFrame.from_records(grid_result.cv_results_['params'])
    param_df['mean_test_nmse'] = np.sqrt(-grid_result.cv_results_['mean_test_nmse'])
    param_df['std_test_nmse'] = np.sqrt(grid_result.cv_results_['std_test_nmse'])
    param_df['mean_test_mae'] = -grid_result.cv_results_['mean_test_mae']
    param_df['std_test_mae'] = -grid_result.cv_results_['std_test_mae']
    param_df['mean_test_r2'] = -grid_result.cv_results_['mean_test_r2']
    param_df['std_test_r2'] = -grid_result.cv_results_['std_test_r2']
    return param_df.sort_values(by=[sorting]).tail(cols)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def mae(y, y_pred):
    return mean_absolute_error(y,y_pred)

def AdjustHigh(sub_train, y_train):
    AdjustedScores = []
    for i in np.arange(.994, 1.000, 0.01):
        for j in np.arange(1.00, 1.10, .01):

            q1 = sub_train['SalePrice'].quantile(0.0025)
            q2 = sub_train['SalePrice'].quantile(0.0045)
            q3 = sub_train['SalePrice'].quantile(i)

            #Verify the cutoffs for the adjustment
            print(q1,q2,q3)
            # adjust at low end
            #sub_train['SalePrice2'] = sub_train['SalePrice'].apply(lambda x: x if x > q1 else x*0.79)
            #sub_train['SalePrice2'] = sub_train['SalePrice'].apply(lambda x: x if x > q2 else x*0.89)

            # adjust at high end
            sub_train['SalePrice2'] = sub_train['SalePrice'].apply(lambda x: x if x < q3 else x*j)

            Predicted = sub_train['SalePrice2']
            Actual = inv_boxcox1p(y_train, lam_l)

            # Pre-adjustment score
            #print("mae for boxcox(SalePrice)",mae(y_train, sub_train['SalePrice2']))
            #print("mse for boxcox(SalePrice)",rmsle(y_train, sub_train['SalePrice2']))
            #print("mae for SalePrice",mae(Actual, Predicted))
            #print("mse for SalePrice",rmsle(Actual, Predicted))

            AdjustedScores.append([i, j, mae(y_train, boxcox1p(sub_train['SalePrice2'], lam_l)), rmsle(y_train, boxcox1p(sub_train['SalePrice2'], lam_l)), mae(Actual, Predicted), rmsle(Actual, Predicted)])

    df_adj = pd.DataFrame(AdjustedScores, columns=["QUANT","COEF","MAE_BC","RMSE_BC","MAE_SP","RMSE_SP"])
    print('quantiles vs coefficients')
    df_adj.sort_values(by=['RMSE_BC'])
    print(df_adj)
    df2 = df_adj.sort_values(by=['RMSE_BC']).reset_index()
    coef_hi = df2.COEF[0]
    quant_hi = df2.QUANT[0]
    return (coef_hi, quant_hi)

def AdjustLow(sub_train, y_train):
    AdjustedScores = []
    for i in np.arange(.00, .02, 0.001):
        for j in np.arange(.90, 1.10, 0.01):

            q1 = sub_train['SalePrice'].quantile(i)
            q2 = sub_train['SalePrice'].quantile(0.1)
            q3 = sub_train['SalePrice'].quantile(.995)

            #Verify the cutoffs for the adjustment
            #print(q1,q2,q3)
            # adjust at low end
            sub_train['SalePrice2'] = sub_train['SalePrice'].apply(lambda x: x if x > q1 else x*j)
            #sub_train['SalePrice2'] = sub_train['SalePrice'].apply(lambda x: x if x > q2 else x*0.89)

            # adjust at high end
            #sub_train['SalePrice2'] = sub_train['SalePrice'].apply(lambda x: x if x < q3 else x*j)

            Predicted = sub_train['SalePrice2']
            Actual = inv_boxcox1p(y_train, lam_l)
            AdjustedScores.append([i, j, mae(y_train, boxcox1p(sub_train['SalePrice2'], lam_l)), rmsle(y_train, boxcox1p(sub_train['SalePrice2'], lam_l)), mae(Actual, Predicted), rmsle(Actual, Predicted)])
  
    df_adj = pd.DataFrame(AdjustedScores, columns=["QUANT","COEF","MAE_BC","RMSE_BC","MAE_SP","RMSE_SP"])
    print('quantiles vs coefficients')
    df_adj.sort_values(by=['RMSE_BC'])
    print(df_adj)
    df2 = df_adj.sort_values(by=['RMSE_BC']).reset_index()
    coef_lo = df2.COEF[0]
    quant_lo = df2.QUANT[0]
    return (coef_lo, quant_lo)

# 
# List of possible scoring values:  
# Regression  
# 
# ‘explained_variance’ metrics.explained_variance_score  
# ‘max_error’ metrics.max_error  
# ‘neg_mean_absolute_error’ metrics.mean_absolute_error  
# ‘neg_mean_squared_error’ metrics.mean_squared_error  
# ‘neg_root_mean_squared_error’ metrics.mean_squared_error  
# ‘neg_mean_squared_log_error’ metrics.mean_squared_log_error  
# ‘neg_median_absolute_error’ metrics.median_absolute_error  
# ‘r2’ metrics.r2_score  
# ‘neg_mean_poisson_deviance’ metrics.mean_poisson_deviance  
# ‘neg_mean_gamma_deviance’ metrics.mean_gamma_deviance  

# -  **LASSO  Regression**  : 
# 
# This model may be very sensitive to outliers. So we need to made it more robust on them. For that we use the sklearn's  **Robustscaler()**  method on pipeline, also want to compare to StandardScaler() => RobustScaler() is slightly better

# In[11]:


# initialize the algorithm for the GridSearchCV function
lasso = Lasso()
tuningLasso = 1 # takes 2 minutes to complete

if (tuningLasso == 1):
    # use this when tuning
    param_grid={
        'alpha':[0.01,], # done, lower keeps getting better, but don't want to go too low and begin overfitting (alpha is related to L1 reg)
        'fit_intercept':[True], # done, big difference
        'normalize':[False], # done, big difference
        'precompute':[False], # done, no difference
        'copy_X':[True], # done, no difference
        'max_iter':[200], # done
        'tol':[0.005], # done, not much difference
        'warm_start':[False], # done, no difference
        'positive':[False], # done, big difference
        'random_state':[1],
        'selection':['cyclic'] # done both are same, cyclic is default
    }

else:
    # use this when not tuning
    param_grid={
        'alpha':[0.2],
        'fit_intercept':[True],
        'normalize':[False],
        'precompute':[False],
        'copy_X':[True],
        'max_iter':[200],
        'tol':[0.0001],
        'warm_start':[False],
        'positive':[False],
        'random_state':[None],
        'selection':['cyclic']
    }

scorers = {
    'r2': 'r2',
    'nmsle': 'neg_mean_squared_log_error', 
    'nmse': 'neg_mean_squared_error',
    'mae': 'neg_mean_absolute_error'
}
# To be used within GridSearch 
inner_cv = KFold(n_splits=4, shuffle=True, random_state=None)

# To be used in outer CV 
outer_cv = KFold(n_splits=4, shuffle=True, random_state=None)    

#inner loop KFold example:
gsc = GridSearchCV(
    estimator=lasso,
    param_grid=param_grid,
    #scoring='neg_mean_squared_error', # 'roc_auc', # or 'r2', etc
    scoring=scorers, # 'roc_auc', # or 'r2', etc -> can output several scores, but only refit to one. Others are just calculated
    #scoring='neg_mean_squared_error', # or look here for other choices 
    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    #cv=5,
    cv=inner_cv, # this will use KFold splitting to change train/test/validation datasets randomly
    verbose=0,
    return_train_score=True, # keep the other scores
    refit='nmse' # use this one for optimizing
)

grid_result = runGSCV(2, train, y_train)

rd = result_details(grid_result,'random_state',100)
rd[['random_state','alpha','mean_test_nmse','std_test_nmse','mean_test_mae','std_test_mae','mean_test_r2','std_test_r2']].sort_values(by=['random_state','alpha'])


# In[12]:


tuning_lasso = 1
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1)) # was 1
lasso_new = make_pipeline(RobustScaler(), Lasso(**grid_result.best_params_))
l = {'alpha': 0.01, 'copy_X': True, 'fit_intercept': True, 'max_iter': 200, 'normalize': False, 'positive': False, 'precompute': False, 'random_state': 1, 'selection': 'cyclic', 'tol': 0.005, 'warm_start': False}
#l = "{'alpha': 0.2, 'copy_X': True, 'fit_intercept': True, 'max_iter': 200, 'normalize': False, 'positive': False, 'precompute': False, 'random_state': 1, 'selection': 'cyclic', 'tol': 0.005, 'warm_start': False}"
#Lasso_new = make_pipeline(RobustScaler(), Lasso(**l))
#lasso_ss = make_pipeline(StandardScaler(), Lasso(alpha =0.0005, random_state=1)) # was 1 => worse score


# In[13]:


if (tuning_lasso == 1):
    #TEMP
    model_results = [] # model flow, mae, rmsle
    models = [lasso, lasso_new]

    for model in models:
        #print(model)
        with MyTimer(): 
            scores = calc_all_scores(model,5,5)
        #print("------------------------------------------")
        model_results.append([model, (-scores['test_mae']).mean(), (np.sqrt(-scores['test_nmse'])).mean(), scores['test_r2'].mean()])

    df_mr = pd.DataFrame(model_results, columns = ['model','mae','rmsle','r2'])
    df_mr.sort_values(by=['rmsle'])


# In[14]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1)) # was 1

if (tuning_lasso == 1):
    for i in [2,5,20,42,99]:
        from sklearn.linear_model import Lasso
        print('random_state =',i)

        l = {'alpha': 0.01, 'copy_X': True, 'fit_intercept': True, 'max_iter': 200, 'normalize': False, 'positive': False, 'precompute': False, 'selection': 'cyclic', 'tol': 0.005, 'warm_start': False}
        lasso_new = make_pipeline(RobustScaler(), Lasso(**l, random_state=i))
        #lasso_new = Lasso(**l, random_state=i)

        model_results = [] # model flow, mae, rmsle
        models = [lasso, lasso_new]

        for model in models:
            #print(model)
            with MyTimer(): 
                scores = calc_all_scores(model,5,5)
            #print("------------------------------------------")
            model_results.append([model, (-scores['test_mae']).mean(), (np.sqrt(-scores['test_nmse'])).mean(), scores['test_r2'].mean()])

        df_mr = pd.DataFrame(model_results, columns = ['model','mae','rmsle','r2'])
        print(df_mr.sort_values(by=['rmsle']))
else:
    lasso_new = make_pipeline(RobustScaler(), Lasso(**grid_result.best_params_, random_state=i))


# In[15]:


model_results


# - **Elastic Net Regression** :
# 
# again made robust to outliers
# 
# combines Lasso L1 Linear regularization and Ridge L2 Quadratic/Squared regularization penalties together into one algorithm

# In[16]:


# initialize the algorithm for the GridSearchCV function
ENet = ElasticNet()
tuningENet = 0 # takes 2 minutes to complete

if (tuningENet == 1):
    # use this when tuning
    param_grid={
        'alpha':[0.005,0.01,0.05,0.1],
        'l1_ratio':[.6,.65,.7,.75,.8,.85,.9],
        'fit_intercept':[True], # ,False
        'normalize':[False], # True,
        'max_iter':range(50,500,50),
        'selection':['random'], # 'cyclic',
        'random_state':[None]
    }

else:
    # use this when not tuning
    param_grid={
        'alpha':[0.05],
        'l1_ratio':[.85],
        'fit_intercept':[True],
        'normalize':[False],
        'max_iter':[100], # default 1000
        'selection':['random']
    }

scorers = {
    'r2': 'r2',
    'nmsle': 'neg_mean_squared_log_error',
    'nmse': 'neg_mean_squared_error',
    'mae': 'neg_mean_absolute_error'
}
# To be used within GridSearch 
inner_cv = KFold(n_splits=4, shuffle=True, random_state=None)

# To be used in outer CV 
outer_cv = KFold(n_splits=4, shuffle=True, random_state=None)    

#inner loop KFold example:
gsc = GridSearchCV(
    estimator=ENet,
    param_grid=param_grid,
    #scoring='neg_mean_squared_error', # 'roc_auc', # or 'r2', etc
    scoring=scorers, # 'roc_auc', # or 'r2', etc -> can output several scores, but only refit to one. Others are just calculated
    #scoring='neg_mean_squared_error', # or look here for other choices 
    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    #cv=5,
    cv=inner_cv, # this will use KFold splitting to change train/test/validation datasets randomly
    verbose=0,
    return_train_score=True, # keep the other scores
    refit='nmse' # use this one for optimizing
)

grid_result = runGSCV(2, train, y_train)

rd = result_details(grid_result,'mean_test_nmse',100)
rd[['alpha','l1_ratio','mean_test_nmse','std_test_nmse','mean_test_mae','std_test_mae','mean_test_r2','std_test_r2']]#.sort_values(by=['n_estimators','mean_test_nmse'])


# In[17]:


#ENet_orig = make_pipeline(StandardScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
ENet = make_pipeline(StandardScaler(), ElasticNet(**grid_result.best_params_, random_state=3))
ENet_new = make_pipeline(RobustScaler(), ElasticNet(**grid_result.best_params_, random_state=3))


# - **Kernel Ridge Regression** :

# In[18]:


tune_kr = 1
if (tune_kr == 1):
    # initialize the algorithm for the GridSearchCV function
    KRR = KernelRidge()
    tuningKRR = 0 # this took 40 mins, 20 per iteration

    if (tuningKRR == 1):
        # use this when tuning
        param_grid={
            'alpha':[2.2,2.4], 
            'kernel':['polynomial'], #for entire list see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.kernel_metrics.html#sklearn.metrics.pairwise.kernel_metrics
            'gamma':[0.0001,0.001,0.01,0.1],
            'degree':[1,2,3,4,5,6], 
            'coef0':[0.1,0.3,0.5,1.0,2.0]
        }

    else:
        # use this when not tuning
        # nmse: Best mae 583416973.611280 using {'alpha': 2.2, 'coef0': 0.5, 'degree': 5, 'gamma': 0.001, 'kernel': 'polynomial'}
        # mae: Best mae 15805.764347 using {'alpha': 2.0, 'coef0': 0.1, 'degree': 5, 'gamma': 0.001, 'kernel': 'polynomial'}
        param_grid={
            'alpha':[2.2], 
            'kernel':['polynomial'], # 'linear', 'rbf'
            'gamma':[0.001],
            'degree':[4], 
            'coef0':[1.0]
        }
    scorers = {
        'r2': 'r2',
        'nmsle': 'neg_mean_squared_log_error',
        'nmse': 'neg_mean_squared_error',
        'mae': 'neg_mean_absolute_error'
    }
    # To be used within GridSearch 
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=None)
    # To be used in outer CV 
    outer_cv = KFold(n_splits=4, shuffle=True, random_state=None)    

    #inner loop KFold example:
    gsc = GridSearchCV(
        estimator=KRR,
        param_grid=param_grid,
        #scoring='neg_mean_squared_error', # 'roc_auc', # or 'r2', etc
        scoring=scorers, # 'roc_auc', # or 'r2', etc -> can output several scores, but only refit to one. Others are just calculated
        #scoring='neg_mean_squared_error', # or look here for other choices 
        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        #cv=5,
        cv=inner_cv, # this will use KFold splitting to change train/test/validation datasets randomly
        verbose=0,
        return_train_score=True, # keep the other scores
        refit='nmse' # use this one for optimizing
    )

    grid_result = runGSCV(2, train, y_train)

    rd = result_details(grid_result,'mean_test_nmse',100)
    rd[['alpha','degree','mean_test_nmse','std_test_nmse','mean_test_mae','std_test_mae','mean_test_r2','std_test_r2']]#.sort_values(by=['n_estimators','mean_test_nmse'])


# In[19]:


#KRR_orig = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

#KRR = KernelRidge(**grid_result.best_params_)
krr = {'alpha': 2.2, 'coef0': 0.5, 'degree': 5, 'gamma': 0.001, 'kernel': 'polynomial'}
KRR = KernelRidge(**krr)
#KRR = KernelRidge(alpha=2.2, coef0=0.5, degree=5, gamma=0.001, kernel='polynomial')

if (tune_kr == 1):
    KRR_new = KernelRidge(**grid_result.best_params_)
else:
    krr_new = {'alpha': 2.3, 'coef0': 1.0, 'degree': 4, 'gamma': 0.001, 'kernel': 'polynomial'}
    KRR_new = KernelRidge(**krr)


# - **Random Forest Regressor** :

# This model needs improvement, will run cross validate on it

# In[20]:


# initialize the algorithm for the GridSearchCV function
rf = RandomForestRegressor()
tuningRF = 0 # this took 2 hours last time, 1 hour per iteration

if (tuningRF == 1):
    # use this when tuning
    param_grid={
        'max_depth':[3,4,5],
        'max_features':[None,'sqrt','log2'], 
        # 'max_features': range(50,401,50),
        # 'max_features': [50,100], # can be list or range or other
        'n_estimators':range(25,100,25), 
        #'class_weight':[None,'balanced'],  
        'min_samples_leaf':range(5,15,5), 
        'min_samples_split':range(10,30,10), 
        'criterion':['mse', 'mae'] 
    }

else:
    # use this when not tuning
    param_grid={
        'max_depth':[5],
        'max_features':[None], # max_features is None is default and works here, removing 'sqrt','log2'
        # 'max_features': range(50,401,50),
        # 'max_features': [50,100], # can be list or range or other
        'n_estimators': [50], # number of trees selecting 100, removing range(50,126,25)
        #'class_weight':[None], # None was selected, removing 'balanced'
        'min_samples_leaf': [5], #selecting 10, removing range 10,40,10)
        'min_samples_split': [10], # selecting 20, removing range(20,80,10),
        'criterion':['mse'] # remove gini as it is never selected
    }

scorers = {
    'r2': 'r2',
    'nmsle': 'neg_mean_squared_log_error',
    'nmse': 'neg_mean_squared_error',
    'mae': 'neg_mean_absolute_error'
}
# To be used within GridSearch 
inner_cv = KFold(n_splits=4, shuffle=True, random_state=None)

# To be used in outer CV 
outer_cv = KFold(n_splits=4, shuffle=True, random_state=None)    

#inner loop KFold example:
gsc = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    #scoring='neg_mean_squared_error', # 'roc_auc', # or 'r2', etc
    scoring=scorers, # 'roc_auc', # or 'r2', etc -> can output several scores, but only refit to one. Others are just calculated
    #scoring='neg_mean_squared_error', # or look here for other choices 
    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    #cv=5,
    cv=inner_cv, # this will use KFold splitting to change train/test/validation datasets randomly
    verbose=0,
    return_train_score=True, # keep the other scores
    refit='nmse' # use this one for optimizing
)

grid_result = runGSCV(2, train, y_train)

rd = result_details(grid_result,'mean_test_nmse',100)
rd[['n_estimators','mean_test_nmse','std_test_nmse','mean_test_mae','std_test_mae','mean_test_r2','std_test_r2']]#.sort_values(by=['n_estimators','mean_test_nmse'])


# In[21]:


#RF_orig = make_pipeline(StandardScaler(), RandomForestRegressor(max_depth=3,n_estimators=500))
RF = make_pipeline(StandardScaler(), RandomForestRegressor(**grid_result.best_params_)) # better than default, but still not good
RF_new = make_pipeline(RobustScaler(), RandomForestRegressor(**grid_result.best_params_)) # better than default, but still not good


# In[22]:


print("Optimize GBoost: ", datetime.datetime.now())


# - **Gradient Boosting Regression** :
# 
# With **huber**  loss that makes it robust to outliers
#     

# Optimize

# In[23]:


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state=5) # was 5
# learning_ratefloat, default=0.1
# learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.
tuning_gb = 0
if (tuning_gb == 1):
    # initialize the algorithm for the GridSearchCV function
    GBoost_new = GradientBoostingRegressor()
    tuningGB = 1
    if (tuningGB == 1):
        # use this when tuning
        param_grid={
            #'loss':['ls','lad','huber','quantile'],
            'loss':['huber'], # done
            'learning_rate':[0.05],
            'n_estimators':[3000], # done
            'subsample':[1.0],
            #'criterion':['friedman_mse','mse','mae'],
            'criterion':['friedman_mse'], # done
            'min_samples_split':[10],
            'min_samples_leaf':[15],
            'min_weight_fraction_leaf':[0.0],
            'max_depth':[2,3,4], # done
            'min_impurity_decrease':[0.0],
            'min_impurity_split':[None],
            'init':[None],
            'random_state':[None],
            #'max_features':[None,'auto','sqrt','log2'],
            'max_features':['sqrt'], # done
            'alpha':[0.60], # done
            'verbose':[0],
            'max_leaf_nodes':[None],
            'warm_start':[False],
            'presort':['deprecated'],
            'validation_fraction':[0.1],
            'n_iter_no_change':[None],
            'tol':[0.0001],
            'ccp_alpha':[0.0],
            'random_state':[5,20,42]
        }
    else:
        # use this when not tuning
        param_grid={
            'loss':['huber'], 
            'learning_rate':[0.05],
            'n_estimators':[3000], 
            'subsample':[1.0],
            'criterion':['friedman_mse'], 
            'min_samples_split':[10],
            'min_samples_leaf':[15],
            'min_weight_fraction_leaf':[0.0],
            'max_depth':[2], 
            'min_impurity_decrease':[0.0],
            'min_impurity_split':[None],
            'init':[None],
            'random_state':[None],
            'max_features':['sqrt'], 
            'alpha':[0.60], 
            'verbose':[0],
            'max_leaf_nodes':[None],
            'warm_start':[False],
            'presort':['deprecated'],
            'validation_fraction':[0.1],
            'n_iter_no_change':[None],
            'tol':[0.0001],
            'ccp_alpha':[0.0],
            'random_state':[5]
        }
    scorers = {
        'r2': 'r2',
        'nmsle': 'neg_mean_squared_log_error',
        'nmse': 'neg_mean_squared_error',
        'mae': 'neg_mean_absolute_error'
    }
    # To be used within GridSearch 
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=None)
    # To be used in outer CV 
    outer_cv = KFold(n_splits=4, shuffle=True, random_state=None)    

    #inner loop KFold example:
    gsc = GridSearchCV(
        estimator=GBoost_new,
        param_grid=param_grid,
        scoring=scorers, # 'roc_auc', # or 'r2', etc -> can output several scores, but only refit to one. Others are just calculated
        cv=inner_cv, # this will use KFold splitting to change train/test/validation datasets randomly
        verbose=0,
        return_train_score=True, # keep the other scores
        refit='nmse' # use this one for optimizing
    )

    grid_result = runGSCV(2, subtrain, y_subtrain)

rd = result_details(grid_result,'mean_test_nmse',100)
rd[['criterion','max_depth','mean_test_nmse','std_test_nmse','mean_test_mae','std_test_mae','mean_test_r2','std_test_r2']]#.sort_values(by=['n_estimators','mean_test_nmse'])


# In[24]:


rd[['criterion','max_depth','mean_test_nmse','std_test_nmse','mean_test_mae','std_test_mae','mean_test_r2','std_test_r2']].sort_values(by=['criterion','max_depth'])


# I am getting conflicting results for  best params, sometimes huber or lad and sometimes friedman_mse or mae, so will look at more detailed output. This style output is much more useful for deciding between parameter values, adding the different random states shows the consistency, or lack of, for each setting

# maybe do a groupby to make this table more manageable and easier to read

# In[25]:


if (tuning_gb == 1):
    GBoost_new = GradientBoostingRegressor(**grid_result.best_params_)
else:
    gbr  = {'alpha': 0.6, 'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.05, 'loss': 'huber', 'max_depth': 3, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 15, 'min_samples_split': 10, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 3000, 'n_iter_no_change': None, 'presort': 'deprecated', 'random_state': 5, 'subsample': 1.0, 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}
    GBoost_new = GradientBoostingRegressor(**gbr)


# In[26]:


if (tuning_gb == 1):
    #TEMP
    model_results = [] # model flow, mae, rmsle
    models = [GBoost, GBoost_new]

    for model in models:
        #print(model)
        with MyTimer(): 
            scores = calc_all_scores(model,5,5)
        #print("------------------------------------------")
        model_results.append([model, (-scores['test_mae']).mean(), (np.sqrt(-scores['test_nmse'])).mean(), scores['test_r2'].mean()])

    df_mr = pd.DataFrame(model_results, columns = ['model','mae','rmsle','r2'])
    df_mr.sort_values(by=['rmsle'])


# In[27]:


if (tuning_gb == 1):
    GBoost.fit(subtrain.values, y_subtrain)
    gboost_train_pred = inv_boxcox1p(GBoost.predict(subtrain.values), lam_l)
    gboost_val_pred = inv_boxcox1p(GBoost.predict(val.values), lam_l)
    print('GBoost')
    print('train results')
    print(mae(y_subtrain, gboost_train_pred))
    print(rmsle(y_subtrain, gboost_train_pred))
    print('test results')
    print(mae(y_val, gboost_val_pred))
    print(rmsle(y_val, gboost_val_pred))

    GBoost_new.fit(subtrain.values, y_subtrain)
    gboost_train_pred = inv_boxcox1p(GBoost_new.predict(subtrain.values), lam_l)
    gboost_val_pred = inv_boxcox1p(GBoost_new.predict(val.values), lam_l)
    print('GBoost_new')
    print('train results')
    print(mae(y_subtrain, gboost_train_pred))
    print(rmsle(y_subtrain, gboost_train_pred))
    print('test results')
    print(mae(y_val, gboost_val_pred))
    print(rmsle(y_val, gboost_val_pred))


# In[28]:


if (tuning_gb == 1):
    GBoost_new.fit(train.values, y_train)
    gboost_test_pred = inv_boxcox1p(GBoost_new.predict(test.values), lam_l)
    gboost_test_pred
    sub = pd.DataFrame()
    #sub['Id'] = test['Id']
    sub['Id'] = test_ID
    sub['SalePrice'] = gboost_test_pred
    sub.to_csv('submission.csv',index=False)


# In[29]:


if (tuning_gb == 1):
    GBoost.fit(train.values, y_train)
    gboost_test_pred = inv_boxcox1p(GBoost.predict(test.values), lam_l)
    gboost_test_pred
    sub = pd.DataFrame()
    #sub['Id'] = test['Id']
    sub['Id'] = test_ID
    sub['SalePrice'] = gboost_test_pred
    sub.to_csv('submission.csv',index=False)


# - **Additional testing**
# 
# Compare any random state

# In[30]:


if (tuning_gb == 1):

    model_results = [] # model flow, mae, rmsle
    models = [GBoost, GBoost_new]

    for model in models:
        #print(model)
        with MyTimer(): 
            scores = calc_all_scores(model,10,10)
        #print("------------------------------------------")
        model_results.append([model, (-scores['test_mae']).mean(), (np.sqrt(-scores['test_nmse'])).mean(), scores['test_r2'].mean()])

    df_mr = pd.DataFrame(model_results, columns = ['model','mae','rmsle','r2'])
    df_mr.sort_values(by=['rmsle'])


# - **XGBoost** :

# In[31]:


print("Optimize XGB: ", datetime.datetime.now())


# In[32]:


tuning_xgb = 0
if (tuning_xgb == 1):
    # initialize the algorithm for the GridSearchCV function# initialize the algorithm for the GridSearchCV function
    xgb1 = xgb.XGBRegressor()
    tuningXGB = 1 # this took 2 hours last time, 1 hour per iteration

    if (tuningXGB == 1):
        # use this when tuning
        param_grid={
            'colsample_bytree':[0.4603],
            'gamma':[0.0468], # done - all values almost identical results
            'colsample_bylevel':[0.3], # done - all give same result
            'objective':['reg:squarederror'], # done - Default:'reg:squarederror', None, reg:pseudohubererror, reg:squaredlogerror, reg:gamma
            'booster':['gbtree'], # done - Default: 'gbtree', 'gblinear' or 'dart'
            'learning_rate':[0.04], # done
            'max_depth':[3], # - done
            'importance_type':['gain'], # done - all give same value, Default:'gain', 'weight', 'cover', 'total_gain' or 'total_cover'
            'min_child_weight':[1.7817], # done - no difference with several values
            'n_estimators':[1000], # done
            'reg_alpha':[0.4], # done
            'reg_lambda':[0.8571], # done
            'subsample':[0.5], # done
            'silent':[1],
            'random_state':[7],
            'scale_pos_weight':[1],
            'eval_metric':['rmse'], # done - all options have same results  Default:rmse for regression rmse, mae, rmsle, logloss, cox-nloglik
            #'nthread ':[-1],
            'verbosity':[0]
        }

    else:
        # use this when not tuning
        param_grid={
            'colsample_bytree':[0.4603],
            'gamma':[0.0468],
            'colsample_bylevel':[0.3],
            'objective':['reg:squarederror'], # 'binary:logistic', 'reg:squarederror', 'rank:pairwise', None
            'booster':['gbtree'], # 'gbtree', 'gblinear' or 'dart'
            'learning_rate':[0.04],
            'max_depth':[3],
            'importance_type':['gain'], # 'gain', 'weight', 'cover', 'total_gain' or 'total_cover'
            'min_child_weight':[1.7817],
            'n_estimators':[1000],
            'reg_alpha':[0.4],
            'reg_lambda':[0.8571],
            'subsample':[0.5],
            'silent':[1],
            'random_state':[7],
            'scale_pos_weight':[1],
            'eval_metric':['rmse'],
            #'nthread ':[-1],
            'nthread ':[-1],
            'verbosity':[0]
        }

    scorers = {
        'r2': 'r2',
        'nmsle': 'neg_mean_squared_log_error',
        'nmse': 'neg_mean_squared_error',
        'mae': 'neg_mean_absolute_error'
    }
    # To be used within GridSearch 
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=None)

    # To be used in outer CV 
    outer_cv = KFold(n_splits=4, shuffle=True, random_state=None)    

    #inner loop KFold example:
    gsc = GridSearchCV(
        estimator=xgb1,
        param_grid=param_grid,
        #scoring='neg_mean_squared_error', # 'roc_auc', # or 'r2', etc
        scoring=scorers, # 'roc_auc', # or 'r2', etc -> can output several scores, but only refit to one. Others are just calculated
        #scoring='neg_mean_squared_error', # or look here for other choices 
        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        #cv=5,
        cv=inner_cv, # this will use KFold splitting to change train/test/validation datasets randomly
        verbose=0,
        return_train_score=True, # keep the other scores
        refit='nmse' # use this one for optimizing
    )

    grid_result = runGSCV(2, train, y_train)


# xgb reference: https://xgboost.readthedocs.io/en/latest/parameter.html

# In[33]:


if (tuning_xgb == 1):
    rd = result_details(grid_result,'mean_test_nmse',100)

    rd[['random_state','eval_metric','mean_test_nmse','std_test_nmse','mean_test_mae','std_test_mae','mean_test_r2','std_test_r2']].sort_values(by=['random_state','mean_test_nmse'])


# In[34]:


model_xgb = xgb.XGBRegressor(booster='gbtree',colsample_bytree=0.4603, gamma=0.0468, # colsample_bylevel, objective, booster (gbtree, gblinear or dart.) # Default: 'gbtree'
                             learning_rate=0.05, max_depth=3, # importance_type (“gain”, “weight”, “cover”, “total_gain” or “total_cover”.)
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state=7) # was random_state=7, cannot set to None 
# maybe an issue with silent=1...

if (tuning_xgb == 1):
    import xgboost as xgb
    for i in [2,5,20,42,99]:
        print('random_state =',i)

        model_xgb_new = xgb.XGBRegressor(booster='gbtree',colsample_bytree=0.4603, gamma=0.0468, # colsample_bylevel, objective, booster (gbtree, gblinear or dart.) # Default: 'gbtree'
                                 learning_rate=0.04, max_depth=3, # importance_type (“gain”, “weight”, “cover”, “total_gain” or “total_cover”.)
                                 min_child_weight=1.7817, n_estimators=1000,
                                 reg_alpha=0.4, reg_lambda=0.8571,
                                 subsample=0.45, silent=1,
                                 random_state=i) # was random_state=7, cannot set to None 

        model_xgb_new2 = xgb.XGBRegressor(booster='gbtree',colsample_bytree=0.4603, gamma=0.0468, # colsample_bylevel, objective, booster (gbtree, gblinear or dart.) # Default: 'gbtree'
                                 learning_rate=0.04, max_depth=3, # importance_type (“gain”, “weight”, “cover”, “total_gain” or “total_cover”.)
                                 min_child_weight=1.7817, n_estimators=1000,
                                 reg_alpha=0.4, reg_lambda=0.8571,
                                 subsample=0.5, silent=1,
                                 random_state=i) # was random_state=7, cannot set to None 

        model_xgb_new3 = xgb.XGBRegressor(booster='gbtree',colsample_bytree=0.4603, gamma=0.0468, # colsample_bylevel, objective, booster (gbtree, gblinear or dart.) # Default: 'gbtree'
                                 learning_rate=0.04, max_depth=3, # importance_type (“gain”, “weight”, “cover”, “total_gain” or “total_cover”.)
                                 min_child_weight=1.7817, n_estimators=1000,
                                 reg_alpha=0.4, reg_lambda=0.8571,
                                 subsample=0.5213, silent=1,
                                 random_state=i) # was random_state=7, cannot set to None

        model_results = [] # model flow, mae, rmsle
        models = [model_xgb, model_xgb_new, model_xgb_new2, model_xgb_new3]

        for model in models:
            #print(model)
            with MyTimer(): 
                scores = calc_all_scores(model,5,5)
            #print("------------------------------------------")
            model_results.append([model, (-scores['test_mae']).mean(), (np.sqrt(-scores['test_nmse'])).mean(), scores['test_r2'].mean()])

        df_mr = pd.DataFrame(model_results, columns = ['model','mae','rmsle','r2'])
        print(df_mr.sort_values(by=['rmsle']))
else:
    model_xgb_new = xgb.XGBRegressor(**grid_result.best_params_)


# Visualize the results

# In[35]:


#import graphviz
#model_xgb.fit(train, y_train,  verbose=False) #  eval_set=[(X_test, y_test)]
#xgb.plot_importance(model_xgb)
#xgb.to_graphviz(model_xgb, num_trees=20)


# In[36]:


print("Optimize LightGBM: ", datetime.datetime.now())


# - **LightGBM** :

# In[37]:


tuning_lgb = 0
if (tuning_lgb == 1):
    # initialize the algorithm for the GridSearchCV function
    #model_lgb = lgb.LGBMRegressor(boosting_type='gbdt', max_depth=- 1, 
    #                            subsample_for_bin=200000, 
    #                            objective='regression',num_leaves=5,
    #                            learning_rate=0.05, n_estimators=720,
    #                            class_weight=None, min_split_gain=0.0, 
    #                            min_child_weight=0.001, min_child_samples=20, subsample=1.0, 
    #                            subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, 
    #                            random_state=None, n_jobs=- 1, silent=True, importance_type='split')

    lgb1 = lgb.LGBMRegressor()
    tuningLGB = 0

    if (tuningLGB == 1):
        # use this when tuning
        param_grid={
            'objective':['regression'], # - only one option for regression
            'boosting_type':['gbdt'], # - done gbdt dart goss rf
            'num_leaves':[5,6], # - done
            'learning_rate':[0.05], # - done
            'n_estimators':[650,750], # - done
            'max_bin':[45,55], # - done
            'bagging_fraction':[0.85], # - done
            'bagging_freq':[5], # - done
            'feature_fraction':[0.2319], # - done
            'feature_fraction_seed':[9], 
            'bagging_seed':[9],
            'min_data_in_leaf':[9], # - done
            'min_sum_hessian_in_leaf':[11], # - done
            'max_depth':[-1], # - -1 means no limit
            'subsample_for_bin':[500,1000], # - done
            'class_weight':[None],
            'min_split_gain':[0.0],
            'min_child_weight':[0.001],
            'min_child_samples':[5], # - done
            'subsample':[1.0],
            'subsample_freq':[0],
            'colsample_bytree':[1.0],
            'reg_alpha':[0.0], # - l1 regularization done
            'reg_lambda':[0.0], # - L2 regularization done
            'random_state':[1],
            'importance_type':['split'] # - done
        }
    else:
        # use this when not tuning
        param_grid={
            'objective':['regression'], # - only one option for regression
            'boosting_type':['gbdt'], # - done gbdt dart goss rf
            'num_leaves':[5], # - done, maybe 5 is okay too
            'learning_rate':[0.05], # - done
            'n_estimators':[650], # - done
            'max_bin':[55], # - done
            'bagging_fraction':[0.85], # - done
            'bagging_freq':[5], # - done
            'feature_fraction':[0.2319], # - done
            'feature_fraction_seed':[9], 
            'bagging_seed':[9],
            'min_data_in_leaf':[9], # - done
            'min_sum_hessian_in_leaf':[11], # - done
            'max_depth':[-1], # - -1 means no limit
            'subsample_for_bin':[1000], # - done
            'class_weight':[None],
            'min_split_gain':[0.0],
            'min_child_weight':[0.001],
            'min_child_samples':[5], # - done
            'subsample':[1.0],
            'subsample_freq':[0],
            'colsample_bytree':[1.0],
            'reg_alpha':[0.0], # - l1 regularization done
            'reg_lambda':[0.0], # - L2 regularization done
            'random_state':[1],
            'importance_type':['split'] # - done
        }

    scorers = {
        'r2': 'r2',
        'nmsle': 'neg_mean_squared_log_error',
        'nmse': 'neg_mean_squared_error',
        'mae': 'neg_mean_absolute_error'
    }
    # To be used within GridSearch 
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=None)

    # To be used in outer CV 
    outer_cv = KFold(n_splits=4, shuffle=True, random_state=None)    

    #inner loop KFold example:
    gsc = GridSearchCV(
        estimator=lgb1,
        param_grid=param_grid,
        #scoring='neg_mean_squared_error', # 'roc_auc', # or 'r2', etc
        scoring=scorers, # 'roc_auc', # or 'r2', etc -> can output several scores, but only refit to one. Others are just calculated
        #scoring='neg_mean_squared_error', # or look here for other choices 
        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        #cv=5,
        cv=inner_cv, # this will use KFold splitting to change train/test/validation datasets randomly
        verbose=0,
        return_train_score=True, # keep the other scores
        refit='nmse' # use this one for optimizing
    )

    grid_result = runGSCV(2, train, y_train)


# In[38]:


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9, random_state=10,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
if (tuning_lgb == 1):
    model_lgb_new = lgb.LGBMRegressor(**grid_result.best_params_)
else:
    lgbm = {'bagging_fraction': 0.85, 'bagging_freq': 5, 'bagging_seed': 9, 'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 1.0, 'feature_fraction': 0.2319, 'feature_fraction_seed': 9, 'importance_type': 'split', 'learning_rate': 0.05, 'max_bin': 55, 'max_depth': -1, 'min_child_samples': 5, 'min_child_weight': 0.001, 'min_data_in_leaf': 9, 'min_split_gain': 0.0, 'min_sum_hessian_in_leaf': 11, 'n_estimators': 650, 'num_leaves': 5, 'objective': 'regression', 'random_state': None, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 1.0, 'subsample_for_bin': 1000, 'subsample_freq': 0}
    model_lgb_new = lgb.LGBMRegressor(**lgbm)


# - **BayesianRidge** :

# In[39]:


# initialize the algorithm for the GridSearchCV function
br = BayesianRidge()
tuningBR = 1 # this took 2 hours last time, 1 hour per iteration

if (tuningBR == 1):
    # use this when tuning
    param_grid={
        'n_iter':[50],
        'tol':[0.001],
        'alpha_1':[1e-06],
        'alpha_2':[1e-05],
        'lambda_1':[1e-05],
        'lambda_2':[1e-06],
        'alpha_init':[None],
        'lambda_init':[None],
        'compute_score':[True,False],
        'fit_intercept':[True,False],
        'normalize':[False,True],
        'copy_X':[True],
        'verbose':[False]
    }

else:
    # use this when not tuning
    param_grid={
        'n_iter':[50],
        'tol':[0.001],
        'alpha_1':[1e-06],
        'alpha_2':[1e-05],
        'lambda_1':[1e-05],
        'lambda_2':[1e-06],
        'alpha_init':[None],
        'lambda_init':[None],
        'compute_score':[True,False],
        'fit_intercept':[True,False],
        'normalize':[False,True],
        'copy_X':[True],
        'verbose':[False]
    }

scorers = {
    'r2': 'r2',
    'nmsle': 'neg_mean_squared_log_error',
    'nmse': 'neg_mean_squared_error',
    'mae': 'neg_mean_absolute_error'
}
# To be used within GridSearch 
inner_cv = KFold(n_splits=4, shuffle=True, random_state=None)

# To be used in outer CV 
outer_cv = KFold(n_splits=4, shuffle=True, random_state=None)    

#inner loop KFold example:
gsc = GridSearchCV(
    estimator=br,
    param_grid=param_grid,
    #scoring='neg_mean_squared_error', # 'roc_auc', # or 'r2', etc
    scoring=scorers, # 'roc_auc', # or 'r2', etc -> can output several scores, but only refit to one. Others are just calculated
    #scoring='neg_mean_squared_error', # or look here for other choices 
    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    #cv=5,
    cv=inner_cv, # this will use KFold splitting to change train/test/validation datasets randomly
    verbose=0,
    return_train_score=True, # keep the other scores
    refit='nmse' # use this one for optimizing
)

grid_result = runGSCV(2, train, y_train)


# In[40]:


tuning_br = 0
BR = BayesianRidge()
if (tuning_br == 1):
    BR_new = BayesianRidge(**grid_result.best_params_)


# In[41]:


rd = result_details(grid_result,'alpha_1',100)
rd[['alpha_1','alpha_2','mean_test_nmse','std_test_nmse','mean_test_mae','std_test_mae','mean_test_r2','std_test_r2']]


# In[42]:


if (tuning_br == 1):
    model_results = [] # model flow, mae, rmsle
    models = [BR, BR_new]

    for model in models:
        #print(model)
        with MyTimer(): 
            scores = calc_all_scores(model,10,10)
        #print("------------------------------------------")
        model_results.append([model, (-scores['test_mae']).mean(), (np.sqrt(-scores['test_nmse'])).mean(), scores['test_r2'].mean()])

    df_mr = pd.DataFrame(model_results, columns = ['model','mae','rmsle','r2'])
    df_mr.sort_values(by=['rmsle'])


# - **More Models** :

# In[43]:


# initialize the algorithm for the GridSearchCV function
# defaults are best
ET = ExtraTreesRegressor()
tuningET = 0 # this took 2 hours last time, 1 hour per iteration

if (tuningET == 1):
    # use this when tuning
    param_grid={
        'n_estimators':[100], # revisit at possibly 2000, but this algorithm becomes really slow for large values
        'criterion':['mse'], # done Default: mse, mae
        'max_depth':[None], # done - above 30 result converges to None
        'min_samples_split':[2], # done - inconsistently better
        'min_samples_leaf':[1],
        'min_weight_fraction_leaf':[0.0],
        'max_features':['auto'], # done - Default:“auto”, “sqrt”, “log2”, None
        'max_leaf_nodes':[None],
        'min_impurity_decrease':[0.0],
        'min_impurity_split':[None],
        'bootstrap':[False],
        'oob_score':[False], # done - True doesn't work, results in a nan value
        'n_jobs':[None],
        'random_state':[1,5,42,55,98],
        'verbose':[0],
        'warm_start':[False],
        'ccp_alpha':[0.0],
        'max_samples':[None] # done - no difference
    }

else:
    # use this when not tuning
    param_grid={
        'n_estimators':[100],
        'criterion':['mse'],
        'max_depth':[None],
        'min_samples_split':[2],
        'min_samples_leaf':[1],
        'min_weight_fraction_leaf':[0.0],
        'max_features':['auto'],
        'max_leaf_nodes':[None],
        'min_impurity_decrease':[0.0],
        'min_impurity_split':[None],
        'bootstrap':[False],
        'oob_score':[False],
        'n_jobs':[None],
        'random_state':[None],
        'verbose':[0],
        'warm_start':[False],
        'ccp_alpha':[0.0],
        'max_samples':[None]
    }

scorers = {
    'r2': 'r2',
    'nmsle': 'neg_mean_squared_log_error',
    'nmse': 'neg_mean_squared_error',
    'mae': 'neg_mean_absolute_error'
}
# To be used within GridSearch 
inner_cv = KFold(n_splits=4, shuffle=True, random_state=None)

# To be used in outer CV 
outer_cv = KFold(n_splits=4, shuffle=True, random_state=None)    

#inner loop KFold example:
gsc = GridSearchCV(
    estimator=ET,
    param_grid=param_grid,
    #scoring='neg_mean_squared_error', # 'roc_auc', # or 'r2', etc
    scoring=scorers, # 'roc_auc', # or 'r2', etc -> can output several scores, but only refit to one. Others are just calculated
    #scoring='neg_mean_squared_error', # or look here for other choices 
    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    #cv=5,
    cv=inner_cv, # this will use KFold splitting to change train/test/validation datasets randomly
    verbose=0,
    return_train_score=True, # keep the other scores
    refit='nmse' # use this one for optimizing
)

grid_result = runGSCV(2, train, y_train)


# In[44]:


rd = result_details(grid_result,'mean_test_nmse',100)
rd[['random_state','n_estimators','mean_test_nmse','std_test_nmse','mean_test_mae','std_test_mae','mean_test_r2','std_test_r2']]


# In[45]:


#ET = make_pipeline(RobustScaler(), ExtraTreesRegressor()) # was 1 Tree algorithms don't need scaling
tuning_et = 0
if (tuning_et == 1):
    for i in [2,5,20,42,99]:
        print('random_state =',i)
        ET = ExtraTreesRegressor(random_state=i)
        #ET2 = make_pipeline(StandardScaler(), ExtraTreesRegressor(random_state=i))

        e = {'bootstrap': False, 'ccp_alpha': 0.0, 'criterion': 'mse', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 6, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 50, 'n_jobs': None, 'oob_score': False, 'verbose': 0, 'warm_start': False}
        ET_new = ExtraTreesRegressor(**e, random_state=i)
        #ET_new2 = make_pipeline(StandardScaler(), ExtraTreesRegressor(**e, random_state=i))

        model_results = [] # model flow, mae, rmsle
        models = [ET, ET_new]

        for model in models:
            #print(model)
            with MyTimer(): 
                scores = calc_all_scores(model,5,5)
            #print("------------------------------------------")
            model_results.append([model, (-scores['test_mae']).mean(), (np.sqrt(-scores['test_nmse'])).mean(), scores['test_r2'].mean()])

        df_mr = pd.DataFrame(model_results, columns = ['model','mae','rmsle','r2'])
        print(df_mr.sort_values(by=['rmsle']))
else:
    ET_new = make_pipeline(RobustScaler(), ExtraTreesRegressor(**grid_result.best_params_))


# In[46]:


# initialize the algorithm for the GridSearchCV function
R = Ridge(alpha=1.0)
tuningR = 1 # this took 2 hours last time, 1 hour per iteration

if (tuningR == 1):
    # use this when tuning
    param_grid={
        'alpha':[8], # done
        'fit_intercept':[True], # done
        'normalize':[False], # done
        'copy_X':[True],
        'max_iter':[None], # done - no difference
        'tol':[0.001],
        'solver':['auto'], # done - Default:‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’
        'random_state':[1,10,42,99,127]
    }

else:
    # use this when not tuning
    param_grid={
        'alpha':[1.0],
        'fit_intercept':[True],
        'normalize':[False],
        'copy_X':[True],
        'max_iter':[None],
        'tol':[0.001],
        'solver':['auto'],
        'random_state':[None]
    }

scorers = {
    'r2': 'r2',
    'nmsle': 'neg_mean_squared_log_error',
    'nmse': 'neg_mean_squared_error',
    'mae': 'neg_mean_absolute_error'
}
# To be used within GridSearch 
inner_cv = KFold(n_splits=4, shuffle=True, random_state=None)

# To be used in outer CV 
outer_cv = KFold(n_splits=4, shuffle=True, random_state=None)    

#inner loop KFold example:
gsc = GridSearchCV(
    estimator=R,
    param_grid=param_grid,
    #scoring='neg_mean_squared_error', # 'roc_auc', # or 'r2', etc
    scoring=scorers, # 'roc_auc', # or 'r2', etc -> can output several scores, but only refit to one. Others are just calculated
    #scoring='neg_mean_squared_error', # or look here for other choices 
    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    #cv=5,
    cv=inner_cv, # this will use KFold splitting to change train/test/validation datasets randomly
    verbose=0,
    return_train_score=True, # keep the other scores
    refit='nmse' # use this one for optimizing
)

grid_result = runGSCV(2, train, y_train)


# In[47]:


rd = result_details(grid_result,'mean_test_nmse',100)
summary = rd[['alpha','random_state','mean_test_nmse','std_test_nmse','mean_test_mae','std_test_mae','mean_test_r2','std_test_r2']].sort_values(by=['random_state','mean_test_nmse'])
#summary.groupby(['fit_intercept'], as_index=False).agg({'mean_test_nmse': 'mean', 'mean_test_nmse': 'std', 'mean_test_mae': 'mean', 'mean_test_r2': 'mean'}).sort_values(by=['mean_test_mae'])
summary.groupby(['alpha'], as_index=False).agg({'mean_test_nmse': 'mean', 'mean_test_mae': 'mean', 'mean_test_r2': 'mean'}).sort_values(by=['mean_test_mae'])


# In[48]:


#R = make_pipeline(RobustScaler(), Ridge(alpha =0.0005, random_state=1)) # was 1
tuning_r = 0
if (tuning_r == 1):
    for i in [2,5,20,42,99]:
        print('random_state =',i)

        r= {'alpha': 8, 'copy_X': True, 'fit_intercept': True, 'max_iter': None, 'normalize': False, 'solver': 'auto', 'tol': 0.001}
        #R_new = make_pipeline(RobustScaler(), Ridge(**r, random_state=i))
        R_new = Ridge(**r, random_state=i)

        model_results = [] # model flow, mae, rmsle
        models = [R, R_new]

        for model in models:
            #print(model)
            with MyTimer(): 
                scores = calc_all_scores(model,5,5)
            #print("------------------------------------------")
            model_results.append([model, (-scores['test_mae']).mean(), (np.sqrt(-scores['test_nmse'])).mean(), scores['test_r2'].mean()])

        df_mr = pd.DataFrame(model_results, columns = ['model','mae','rmsle','r2'])
        print(df_mr.sort_values(by=['rmsle']))
else:
    #R_new = make_pipeline(RobustScaler(), Ridge(**grid_result.best_params_))
    R_new = Ridge(**grid_result.best_params_)


# - **CatBoost**
# 
# Reference: https://catboost.ai/docs/concepts/python-reference_parameters-list.html

# In[49]:


#CBoost = CatBoostRegressor(logging_level='Silent',random_seed=0)
CBoost = CatBoostRegressor(logging_level='Silent', random_state=0, depth=6, n_estimators=1000,eval_metric='RMSE',bagging_temperature=1,grow_policy='SymmetricTree',bootstrap_type='MVS') # l2_leaf_reg, learning_rate unknown


# In[50]:


tuning_cb = 0
if (tuning_cb == 1):
    # initialize the algorithm for the GridSearchCV function
    cb = CatBoostRegressor()
    if (tuningCB == 1):
        # use this when tuning
        param_grid={
            'nan_mode':['Min'],
            'eval_metric':['RMSE'],
            'iterations':[1000],
            'sampling_frequency':['PerTree'],
            'leaf_estimation_method':['Newton'],
            'grow_policy':['SymmetricTree'],
            'penalties_coefficient':[1],
            'boosting_type':['Plain'],
            'model_shrink_mode':['Constant'],
            'feature_border_type':['GreedyLogSum'],
            #'bayesian_matrix_reg':[0.10000000149011612],
            'l2_leaf_reg':[3],
            'random_strength':[1],
            'rsm':[1],
            'boost_from_average':[True],
            'model_size_reg':[0.5],
            'subsample':[0.800000011920929],
            'use_best_model':[False],
            'random_seed':[0,2,15],
            'depth':[6], # done
            'border_count':[254],
            #'classes_count':[0],
            #'auto_class_weights':['None'],
            'sparse_features_conflict_fraction':[0],
            'leaf_estimation_backtracking':['AnyImprovement'],
            'best_model_min_trees':[1],
            'model_shrink_rate':[0],
            'min_data_in_leaf':[1],
            'loss_function':['RMSE'],
            'learning_rate':[0.04174000024795532],
            'score_function':['Cosine'],
            'task_type':['CPU'],
            'leaf_estimation_iterations':[1],
            'bootstrap_type':['MVS'],
            'max_leaves':[31],
            'logging_level':['Silent']
        }

    else:
        # use this when not tuning
        param_grid={
            'nan_mode':['Min'],
            'eval_metric':['RMSE'],
            'iterations':[4000],
            'sampling_frequency':['PerTree'],
            'leaf_estimation_method':['Newton'],
            'grow_policy':['SymmetricTree'],
            'penalties_coefficient':[1],
            'boosting_type':['Plain'],
            'model_shrink_mode':['Constant'],
            'feature_border_type':['GreedyLogSum'],
            #'bayesian_matrix_reg':[0.10000000149011612],
            'l2_leaf_reg':[3],
            'random_strength':[1],
            'rsm':[1],
            'boost_from_average':[True],
            'model_size_reg':[0.5],
            'subsample':[0.800000011920929],
            'use_best_model':[False],
            'random_seed':[0],
            'depth':[6],
            'border_count':[254],
            #'classes_count':[0],
            #'auto_class_weights':['None'],
            'sparse_features_conflict_fraction':[0],
            'leaf_estimation_backtracking':['AnyImprovement'],
            'best_model_min_trees':[1],
            'model_shrink_rate':[0],
            'min_data_in_leaf':[1],
            'loss_function':['RMSE'],
            'learning_rate':[0.04174000024795532],
            'score_function':['Cosine'],
            'task_type':['CPU'],
            'leaf_estimation_iterations':[1],
            'bootstrap_type':['MVS'],
            'max_leaves':[31],
            'logging_level':['Silent']
        }

    scorers = {
        'r2': 'r2',
        'nmsle': 'neg_mean_squared_log_error',
        'nmse': 'neg_mean_squared_error',
        'mae': 'neg_mean_absolute_error'
    }
    # To be used within GridSearch 
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=None)

    # To be used in outer CV 
    outer_cv = KFold(n_splits=4, shuffle=True, random_state=None)    

    #inner loop KFold example:
    gsc = GridSearchCV(
        estimator=cb,
        param_grid=param_grid,
        #scoring='neg_mean_squared_error', # 'roc_auc', # or 'r2', etc
        scoring=scorers, # 'roc_auc', # or 'r2', etc -> can output several scores, but only refit to one. Others are just calculated
        #scoring='neg_mean_squared_error', # or look here for other choices 
        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        #cv=5,
        cv=inner_cv, # this will use KFold splitting to change train/test/validation datasets randomly
        verbose=0,
        return_train_score=True, # keep the other scores
        refit='nmse'#, # use this one for optimizing
        #plot=True
    )
    grid_result = runGSCV(2, train, y_train)
    rd = result_details(grid_result,'depth',100)
    rd[['random_state','depth','mean_test_nmse','std_test_nmse','mean_test_mae','std_test_mae','mean_test_r2','std_test_r2']]


# In[51]:


tuning_cb = 0

#R = make_pipeline(RobustScaler(), Ridge(alpha =0.0005, random_state=1)) # was 1
if (tuning_cb == 1):
    for i in [2,5,20,42,99]:
        print('random_state =', i)
        CBoost_new = CatBoostRegressor(**grid_result.best_params_) 

        model_results = [] # model flow, mae, rmsle
        models = [CBoost, CBoost_new]

        for model in models:
            #print(model)
            with MyTimer(): 
                scores = calc_all_scores(model,5,5)
            model_results.append([model.get_params(), (-scores['test_mae']).mean(), (np.sqrt(-scores['test_nmse'])).mean(), scores['test_r2'].mean()])

        df_mr = pd.DataFrame(model_results, columns = ['model','mae','rmsle','r2'])
        print(df_mr.sort_values(by=['rmsle']))
else:
    #R_new = make_pipeline(RobustScaler(), Ridge(**grid_result.best_params_))
    CBoost_new = CatBoostRegressor(logging_level='Silent', random_state=15, depth=5, l2_leaf_reg=1.0, n_estimators=1700,eval_metric='RMSE',learning_rate=0.025,random_strength=3.7,bagging_temperature=1.0,grow_policy='SymmetricTree',bootstrap_type='Bayesian')#,bayesian_matrix_reg=0.10000000149011612)


# In[54]:


AB = AdaBoostRegressor()
from sklearn.svm import SVR
SVR = SVR()
DT = DecisionTreeRegressor()
KN = KNeighborsRegressor()
B = BaggingRegressor()


# ###Base models scores

# Let's see how these base models perform on the data by evaluating the  cross-validation rmsle error

# In[55]:


if (tuning_gb == 1):
    model_results = [] # model flow, mae, rmsle
    models = [GBoost, GBoost_new]# model_lgb_new, lasso_ns, 

    for model in models:
        #print(model)
        with MyTimer(): 
            scores = calc_all_scores(model)
        #print("------------------------------------------")
        model_results.append([model, (-scores['test_mae']).mean(), (np.sqrt(-scores['test_nmse'])).mean(), scores['test_r2'].mean()])

    df_mr = pd.DataFrame(model_results, columns = ['model','mae','rmsle','r2'])
    df_mr.sort_values(by=['rmsle'])


# Calculate Metrics for model list

# In[56]:


print("Calculate Metrics: ", datetime.datetime.now())


# In[57]:


compare_models = 0
if (compare_models == 1):
    model_results = [] # model flow, mae, rmsle
    # GBoost_new is better than GBoost, but has lower final score, think this may be  overfitting
    # BR_new has same results, here, but better final score
    models = [lasso, lasso_new, model_lgb, ENet, ENet_new, KRR, GBoost, GBoost_new, model_xgb, BR, ET, ET_new, RF, RF_new, AB, SVR, DT, KN, B, R, R_new, CBoost, CBoost_new] # worse or same: BR_new, model_lgb_new, lasso_ns, model_xgb_new,

    for model in models:
        #print(model)
        with MyTimer(): 
            scores = calc_all_scores(model)
        #print("------------------------------------------")
        try:
            print(model)
            label = model
        except KeyError as err:
            print("KeyError error: {0}".format(err))
            label = model.__class__()
        except Exception as e:
            print(e.message, e.args)
            label = model.__class__()
        finally:
            print("Continue") 
        print('label', label)    
        #model_results.append([model, (-scores['test_mae']).mean(), (np.sqrt(-scores['test_nmse'])).mean(), scores['test_r2'].mean()])
        model_results.append([label, (-scores['test_mae']).mean(), (np.sqrt(-scores['test_nmse'])).mean(), scores['test_r2'].mean()])

    df_mr = pd.DataFrame(model_results, columns = ['model','mae','rmsle','r2'])
    df_mr.sort_values(by=['rmsle'])


# In[58]:


df_mr.sort_values(by=['rmsle'])


# In[59]:


print("Stacking and Ensembling: ", datetime.datetime.now())


# ##Stacking  models

# ###Simplest Stacking approach : Averaging base models

# We begin with this simple approach of averaging base models.  We build a new **class**  to extend scikit-learn BaseEstimator, RegressorMixin, TransformerMixin classes with our model and also to leverage encapsulation and code reuse ([inheritance][1]) 
# 
# 
#   [1]: https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)

# **Averaged base models class**
# 
# In averaging methods, the driving principle is to build several estimators independently and then to average their predictions. On average, the combined estimator is usually better than any of the single base estimator because its variance is reduced.

# In[60]:


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them (can try median also)
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        # IDEA: return weighted means
        return np.mean(predictions, axis=1)

from sklearn.ensemble import VotingRegressor
import subprocess
polynomial_features = PolynomialFeatures(degree=2, include_bias=False)
from sklearn.feature_selection import f_regression, f_classif

from keras.optimizers import Adam, SGD, RMSprop 
# define base model
#from keras.optimizers import Adam   #for adam optimizer
def baseline_model(opt_sel="adam", learning_rate = 0.001, neurons = 1, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, decay = 0.0002, momentum=0.9):
    def bm():
        #from keras.optimizers import Adam, SGD, RMSprop   #for adam optimizer
        # create model
        model = Sequential()
        model.add(Dense(neurons, input_dim=223, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))

        #model.add(Dense(64, kernel_initializer='normal', activation='relu')),
        #model.add(Dense(13, kernel_initializer='normal', activation='relu')),
        #model.add(Dense(6, kernel_initializer='normal', activation='relu'))

        #opt = SGD(learning_rate=lr_schedule)
        
        model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        if (opt_sel == "adam"):
            opt = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad)
        elif(opt_sel == "sgd"):
            opt = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay, nesterov=True)

        #opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
        #opt = SGD(learning_rate=learning_rate, momentum=0.9) 
        #opt = RMSprop(learning_rate=learning_rate)

        model.compile(loss='mean_squared_error', optimizer=opt)
        return model
    return bm

# for more repeatable results
# ints = tf.random.stateless_uniform([10], seed=(2, 3), minval=None, maxval=None, dtype=tf.int32)
# also
# step1: fix the numpy random seed at the top of code
# step2: be sure that model.fit(shuffle=False)

# dnn = KerasRegressor(build_fn=baseline_model(0.005, 8), epochs=100, batch_size=1, verbose=0)
#for i in range(0,30):

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras import regularizers
from keras import backend as K
from keras.layers import Conv1D
from keras.layers import BatchNormalization
from keras.layers import MaxPool1D
from keras.layers import Flatten
from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import LearningRateScheduler, EarlyStopping

def step_decay(epoch, lr):
    drop = 0.999 # was .999
    epochs_drop = 175.0 # was 175, sgd likes 200+, adam likes 100
    lrate = lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    print("epoch=" + str(epoch) + " lr=" + str(lr) + " lrate=" + str(lrate))
    return lrate


norm = MinMaxScaler().fit(train)
train_norm = norm.transform(train)
test_norm = norm.transform(test)

for i1 in [0,1]: #np.arange(1, 6, 1):
    for j1 in [1000]:
        for k1 in [2]:
            # evaluate model
            optimizer = "sgd"
            lrate = LearningRateScheduler(step_decay)
            early_stopping = EarlyStopping(monitor='val_loss', patience=50, mode='auto', restore_best_weights = True)
            if (optimizer == "adam"):
                model = KerasRegressor(build_fn=baseline_model(opt_sel="adam", learning_rate = 0.001, neurons = 8, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False), epochs=1000, batch_size=2, verbose=1)
            elif (optimizer == "sgd"): # loss = 6.7 with scale,   with norm
                model = KerasRegressor(build_fn=baseline_model(opt_sel="sgd",learning_rate=0.000005,neurons=32,decay=0.000001,momentum=0.9), epochs=1000, batch_size=8, verbose=1)
            callbacks_list = [lrate, early_stopping]    

            #model = Pipeline(steps=[('standardscaler', StandardScaler()),('elasticnet',ElasticNet(alpha=0.01, l1_ratio=0.9, max_iter=350,random_state=3, selection='random', tol=j1))])

            #estimator_list = [('CBoost', CBoost_new),('xgb',model_xgb),('ENet',ENet),('gboost',GBoost),('krr',KRR),('br',BR)]
            #averaged_models = VotingRegressor(estimators=estimator_list, weights=[1,1,1,1,1,1]) 
            estimator_list = [('model', model)]
            weight_list = [1]
            use_averaged = 0
            if (use_averaged == 1):
                averaged_models = VotingRegressor(estimators=estimator_list, weights=weight_list) 
            else:
                averaged_models = model

            hist = averaged_models.fit(train_norm, y_train, shuffle=True, validation_split=0.3, callbacks=callbacks_list)
            num_epochs = len(hist.history['loss'])
            train_loss = hist.history['loss']
            val_loss = hist.history['val_loss']
            train_lr   = hist.history['lr']
            xc         = range(num_epochs)
            plt.figure()
            plt.plot(xc, train_loss, label='train')
            plt.plot(xc, val_loss, label='val')
            plt.plot(xc, train_lr, label='lr')
            plt.show()
            submit_prediction = 1
            averaged_train_pred = averaged_models.predict(train_norm)
            averaged_pred = inv_boxcox1p(averaged_models.predict(test_norm), lam_l)

            sub_train = pd.DataFrame()
            sub_train['Id'] = train_ID
            sub_train['SalePrice'] = inv_boxcox1p(averaged_train_pred, lam_l)

            ch, qh = AdjustHigh(sub_train, y_train)
            cl, ql = AdjustLow(sub_train, y_train)

            sub = pd.DataFrame()
            sub['Id'] = test_ID
            sub['SalePrice'] = averaged_pred

            q1 = sub['SalePrice'].quantile(ql)
            q2 = sub['SalePrice'].quantile(0.1)
            q3 = sub['SalePrice'].quantile(qh)

            # adjust high predictions only
            sub['SalePrice'] = sub['SalePrice'].apply(lambda x: x if (x < q3) else x*ch)
            print(sub.head(10))
            sub.to_csv('submission.csv',index=False)
            #message = "DNN/Adam: seeds np=" + str(numpy_seed) + " tf=" + str(tf_seed) + " rn=" + str(random_seed)
            message = "DNN/norm/" + optimizer + " epochs=" + str(num_epochs) + " batch_size=" + str(k1)
            if (submit_prediction == 1):
                subprocess.run([ "kaggle", "competitions", "submit", "-c", "home-data-for-ml-course", "-f", "submission.csv", "-m", message ])

print("finished")
