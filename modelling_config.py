import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV

from utils import get_configured_logger

logger = get_configured_logger(__name__)


def stratified_sampling(df: pd.DataFrame, target: pd.Series) -> tuple:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.7, random_state=0)
    sss.get_n_splits(df, target)
    for train_index, test_index in sss.split(df, target):
        x_train, x_test = df.loc[train_index, :], df.loc[test_index, :]
        y_train, y_test = target[train_index], target[test_index]
    return x_train, x_test, y_train, y_test


def random_forest_model() -> object:
    rf = RandomForestClassifier(
        random_state=1,
        criterion="gini",
        max_depth=10,
        max_features="auto",
        n_estimators=250,
        min_samples_leaf=30,
        class_weight="balanced",
        n_jobs=-1,
    )
    return rf


def random_forest_tuning(x_train: pd.DataFrame, y_train: pd.DataFrame) -> object:
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [8, 10, 12],
        "max_features": ["auto"],
        "n_estimators": [200, 250, 300],
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)
    return cv_rfc


def feature_importance(rf: object, x_train: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(
        rf.feature_importances_, index=x_train.columns, columns=["importance"]
    ).sort_values("importance", ascending=False)
    return df


def model_metrics(features_df: pd.DataFrame, target: pd.Series, clf: object):
    """
    returns the following results
    precision score, recall score, f1 score
    Accuracy, Gini and AUC for train and test
    of a binary classified model
    Args:
        features_df (pd.DataFrame): predictor variables
        target (pd.Series): target values
        clf (object): ML method
    Returns: print out of results in console
    """
    predictions = np.array(clf.predict(features_df))
    probabilities = clf.predict_proba(features_df)[:, 1]
    logger.info(
        (
            " Confusion matrix on train:\n",
            pd.crosstab(
                target, predictions, rownames=["Actual "], colnames=["Predicted "]
            ),
        )
    )
    for metric in [precision_score, recall_score, f1_score]:
        logger.info(metric.__name__ + f": {metric(target.values, predictions):.2f}")
    logger.info(
        "average precision: {:.2f}".format(
            average_precision_score(target.values, predictions, average="weighted")
        )
    )
    logger.info(
        (" Accuracy: {:.2f}".format(metrics.accuracy_score(target, predictions)))
    )
    # AUC and Gini
    fpr, tpr, thresholds = metrics.roc_curve(target, probabilities, pos_label=1)
    logger.info(f" AUC: {metrics.auc(fpr, tpr): .2f}")
    logger.info(f" Gini: {2 * metrics.auc(fpr, tpr) - 1: .2f}")
    logger.info("End of Model Metric Results")


def model_metrics_print(features_df: pd.DataFrame, target: pd.Series, clf: object):
    """
    returns the following results
    precision score, recall score, f1 score
    Accuracy, Gini and AUC for train and test
    of a binary classified model
    Args:
        features_df (pd.DataFrame): predictor variables
        target (pd.Series): target values
        clf (object): ML method
    Returns: print out of results for notebook
    """
    # predictions = np.array(
    #     [1 if y >= 0.35 else 0 for y in clf.predict_proba(features_df)[:, 1]]
    # )
    predictions = clf.predict(features_df)
    probabilities = clf.predict_proba(features_df)[:, 1]
    print(
        " Confusion matrix on train:\n",
        pd.crosstab(
            target,
            predictions,
            rownames=["Actual Bankrupt "],
            colnames=["Predicted Bankrupt"],
        ),
    )
    for metric in [precision_score, recall_score, f1_score]:
        print(metric.__name__ + f": {metric(target.values, predictions):.2f}")
    print(
        "average precision: {:.2f}".format(
            average_precision_score(target.values, predictions, average="weighted")
        )
    )
    print(f" Accuracy: {metrics.accuracy_score(target, predictions):.2f}")
    # AUC and Gini
    fpr, tpr, thresholds = metrics.roc_curve(target, probabilities, pos_label=1)
    print(f" AUC: {metrics.auc(fpr, tpr): .2f}")
    print(f" Gini: {(2 * metrics.auc(fpr, tpr) - 1): .2f}")
