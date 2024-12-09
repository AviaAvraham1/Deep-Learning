import numpy as np
import sklearn
from pandas import DataFrame
from typing import List
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils import check_array
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, "weights_")
        print(X.shape)
        # Calculate the model prediction, y_pred

        y_pred = np.dot(X, self.weights_)


        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)

        #  Calculate the optimal weights using the closed-form solution you derived.
        #  Use only numpy functions. Don't forget regularization!

        # n_features = X.shape[1]
        # regularization_matrix = np.eye(n_features) # init identity matrix
        # regularization_matrix[0, 0] = 0  # don't regularize the bias

        # XtX = np.dot(X.T, X)
        # XtX_reg = XtX + self.reg_lambda * regularization_matrix
        # XtX_reg_inv = np.linalg.inv(XtX_reg) # inverse 
        # Xty = np.dot(X.T, y)
        # w_opt = np.dot(XtX_reg_inv, Xty)
        num_samples = X.shape[0]
        num_features = X.shape[1]
        lambda_I = self.reg_lambda * num_samples * np.eye(num_features)
        lambda_I[0, 0] = 0  # don't regularize the bias

        # the closed form solution is:
        # w_opt = (XtX + lambda*N*I)^(-1) * Xt * y
        w_opt = np.linalg.inv(np.dot(X.T, X) + lambda_I).dot(X.T).dot(y)

        self.weights_ = w_opt
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


def fit_predict_dataframe(
    model, df: DataFrame, target_name: str, feature_names: List[str] = None,
):
    """
    Calculates model predictions on a dataframe, optionally with only a subset of
    the features (columns).
    :param model: An sklearn model. Must implement fit_predict().
    :param df: A dataframe. Columns are assumed to be features. One of the columns
        should be the target variable.
    :param target_name: Name of target variable.
    :param feature_names: Names of features to use. Can be None, in which case all
        features are used.
    :return: A vector of predictions, y_pred.
    """
    # all features are used if feature_names is None
    if feature_names is None:
        feature_names = df.columns[df.columns != target_name]

    X = df[feature_names].values
    y = df[target_name].values
    
    #print(feature_names)
    # we'd want to train based on the other features, according to the target feature
    y_pred = model.fit_predict(X, y)

    return y_pred



class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray):
        """
        :param X: A tensor of shape (N,D) where N is the batch size and D is
        the number of features.
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X, ensure_2d=True)

        # TODO:
        #  Add bias term to X as the first feature.
        #  See np.hstack().

        xb = np.hstack((np.ones((X.shape[0], 1)), X))

        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """

    def __init__(self, degree=2):
        self.degree = degree
        # You can initialize any additional parameters here if needed
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)

        #  Transform the features of X into new features in X_transformed
        #  Note: You CAN count on the order of features in the Boston dataset
        #  (this class is "Boston-specific"). For example X[:,1] is the second
        #  feature ('ZN').

        # CHAS feature is index 3
        X_modified = np.delete(X, 3, axis=1)

        # CRIM is index 0
        # LSTAT is index 11 after removal of CHAS
        X_modified[:, 0] = np.log1p(X_modified[:, 0])  
        X_modified[:, -1] = np.log1p(X_modified[:, -1])  

        X_transformed = self.poly.fit_transform(X_modified)

        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """

    # TODO: Calculate correlations with target and sort features by 
    
    # get correlation matrix
    corr_matrix = df.corr()

    # take the target feature row from the  matrix, and remove the target feature itself from it
    correlations = corr_matrix[target_feature]
    correlations = correlations.drop(target_feature)

    # get top n correlations in descending order
    top_n = correlations.abs().sort_values(ascending=False).head(n)
    top_n_features = top_n.index
    top_n_corr = correlations[top_n.index]

    return top_n_features, top_n_corr


def mse_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes Mean Squared Error.
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: MSE score.
    """

    # TODO: Implement MSE using numpy.
    mse = np.mean((y - y_pred) ** 2)
    return mse


def r2_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes R^2 score,
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: R^2 score.
    """

    # compute the residual sum of squares
    residuals_sum = np.sum((y - y_pred) ** 2)

    # Compute the total sum of squares
    diff_from_avg_sum = np.sum((y - np.mean(y)) ** 2)

    # Compute R^2
    r2 = 1 - (residuals_sum / diff_from_avg_sum)
    return r2


def cv_best_hyperparams(
    model: BaseEstimator, X, y, k_folds, degree_range, lambda_range
):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    # TODO: Do K-fold cross validation to find the best hyperparameters
    #  Notes:
    #  - You can implement it yourself or use the built in sklearn utilities
    #    (recommended). See the docs for the sklearn.model_selection package
    #    http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    #  - If your model has more hyperparameters (not just lambda and degree)
    #    you should add them to the search.
    #  - Use get_params() on your model to see what hyperparameters is has
    #    and their names. The parameters dict you return should use the same
    #    names as keys.
    #  - You can use MSE or R^2 as a score.

    param_grid = {
        'bostonfeaturestransformer__degree': degree_range,
        'linearregressor__reg_lambda': lambda_range
    }

    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=k_folds,
        scoring=scorer,
        n_jobs=-1,  
    )

    grid_search.fit(X, y)

    best_params = grid_search.best_params_

    return best_params
