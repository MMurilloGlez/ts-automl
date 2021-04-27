"""Feature selection methods for time series forecasting.


Module containing different feature selection models to simplify and decrease
processing time for the prediction models in the library. All wrapped in 
sklearn wrappers so they ¡can be implemented into pipelines.

Parameters
----------

preprocessor: str
    Type of feature selection to use in the prediction. i.e. PCA, Tree
data: pd.Dataframe
    Dataframe containing the full feature set and lags as well as the base time
    series for the training data
n_feat: int
    Number of features to preserve from the dataframe

Returns
-------
pd.Dataframe
    Dataframe containing the n_feat most important features for the prediction, 
    on which to train and apply the forecasting models

"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.decomposition import PCA


def feature_selection_tree(X, y, num_features):
    """
    Feature selection using a LightGBM untrimmed model as a method.

    Fit a generic LightGBM regressor model to X and y training data, then use 
    its feature_importance parameters to select the principal components to keep

    Parameters
    ----------
    X: pd.Dataframe
        Feature set of the training data for the model
    y: pd.Dataframe
        Target variable for the training data for the model.
    num_features: int
        number of features to preserve from the input set.

    Returns
    -------
    list of str
        Column names of the most important features to preserve for the model.

    """

    clf = lgb.LGBMRegressor(n_estimators=40).fit(X, y)
    best_indx_col = clf.feature_importances_.argsort()[-num_features:]
    best_features = list(X_train.columns[best_indx_col])
    
    def order_by_other_list(list_order, list_tobe_order):
        d = {k:v for v,k in enumerate(list_order)}
        list_tobe_order.sort(key=d.get)
        return list_tobe_order
    
    return order_by_other_list(X_train.columns, best_features)


def feature_selection_pca(X, num_features):
    """
    Principal component selection using principal Component Analysis

    Applies sklearn PCA to all of the data (train and test) to reduce the 
    complexity of the problem.

    Parameters
    ----------
    X: pd.Dataframe
        Full dataset features
    num_features: int
        Number of principal components to keep from the data.
    
    Returns
    -------
    pd.Dataframe:
    X data comprised of n principal componentes
    """
    clf = PCA(n_components=num_features).fit(X)
    return clf

