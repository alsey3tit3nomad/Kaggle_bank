from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from sklearn import set_config
set_config(transform_output='pandas')

class MyLog1p(BaseEstimator, TransformerMixin):
    def __init__(self, eps=1e-10):
        self.eps = eps
        self.columns = None
        self.min_ = None

    def fit(self, X, y=None):
        data = X
        self.columns = X.columns
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=self.columns)
        self.min_ = data.min()
        return self

    def transform(self, X, y=None):
        data = X
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=self.columns)
        data = data - self.min_
        data = np.clip(data, a_min=1e-10, a_max=None)
        data = np.log1p(data)
        return pd.DataFrame(data, columns=self.columns)

class Percentile_transform(BaseEstimator, TransformerMixin):
    def __init__(self, q=99.5, columns=None):
        self.q = q
        self.thresholds_ = {}
        self.columns = columns

    def fit(self, X, y=None):
        if (not self.columns):
            self.columns = X.select_dtypes(['number']).columns.tolist()
        for i in self.columns:
            self.thresholds_[i] = np.percentile(X[i], q=self.q)
        return self

    def transform(self, X, y=None):
        if not self.thresholds_ or not self.columns:
            raise ValueError("Сначала  fit")
        data = X.copy()
        for column, threshold in self.thresholds_.items():
            data[f'threshold_{column}'] = data[column] < threshold

        return data



class Percentile_filter(BaseEstimator, TransformerMixin):
    def __init__(self, q=99.5, columns=None):
        self.q = q
        self.thresholds_ = {}
        self.columns = columns

    def fit(self, X, y=None):
        if (not self.columns):
            self.columns = X.select_dtypes(['number']).columns.tolist()
        for i in self.columns:
            self.thresholds_[i] = np.percentile(X[i], q=self.q)
        mask = pd.Series([True for _ in range(X.shape[0])], index=X.index)
        for column, threshold in self.thresholds_.items():
            mask &= (X[column] <= threshold)
        if y is None:
            X = X[mask]  # только X
        else:
            X = X[mask]
            y = y[mask.values]
        return self

    def transform(self, X, y=None):
        if not self.thresholds_:
            raise ValueError("Сначала  fit")
        # mask = pd.Series([True for _ in range(X.shape[0])], index=X.index)
        # for column, threshold in self.thresholds_.items():
        #     mask &= (X[column] <= threshold)
        if y is None:
            return X  # только X
        else:
            return X, y



def create_timestamp(X, y=None):
    month_to_int = {'jan' : 1, 'feb' : 2, 'mar' : 3, 'apr' : 4, 'may' : 5, 'jun' : 6, 'jul' : 7, 'aug' : 8, 'sep' : 9, 'oct' : 10, 'nov' : 11, 'dec' : 12}
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=['day', 'month'])
    month_nums = X['month'].apply(lambda x: month_to_int[x])
    dates = pd.to_datetime({'year' : 1970, 'month' : month_nums, 'day' : X['day']}, errors='coerce')
    start_timestamp = pd.Timestamp('1970-01-01')
    dates = (dates - start_timestamp).dt.days
    return pd.DataFrame(dates, columns=['timestamp'])


from sklearn.preprocessing import FunctionTransformer

timestamp_transformer = FunctionTransformer(create_timestamp, validate=False)

class MyOHE(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        # важно: сразу задаём sparse_output=False
        self.encoder = OneHotEncoder(**kwargs)
        self.columns = None

    def fit(self, X, y=None):
        self.encoder.fit(X)
        self.columns = self.encoder.get_feature_names_out(X.columns)
        return self

    def transform(self, X):
        arr = self.encoder.transform(X)
        # здесь уже точно dense, потому что sparse_output=False
        return pd.DataFrame(arr, columns=self.columns, index=X.index)

    def inverse_transform(self, X):
        return self.encoder.inverse_transform(X)

class FinishPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, pipe_line, cat_features_names):
        self.pipe_line = pipe_line
        self.cat_features_names = cat_features_names

    def fit(self, X, y=None):
        self.pipe_line.fit(X, y)
        return self

    def transform(self, X, y=None):
        X_pr = self.pipe_line.transform(X)
        ohe = self.pipe_line.named_steps['columns_transformer']\
                                .named_transformers_['ohe']\

        new_columns = ohe.get_feature_names_out()
        X_ohe = X_pr[new_columns]
        X_old_cat = ohe.inverse_transform(X_ohe)
        X_old_cat = pd.DataFrame(X_old_cat, columns=self.cat_features_names, index=X_ohe.index)
        X_pr = X_pr.drop(columns=new_columns).join(X_old_cat)
        if y is not None:
            y_new = y.loc[X_pr.index]
            return X_pr, y_new
        return X_pr

    def fit_transform(self, X, y=None, **fit_params):
        X_pr = super().fit_transform(X, y, **fit_params)
        if y is not None:
            y_new = y.loc[X_pr.index]
            return X_pr, y_new
        return X_pr
