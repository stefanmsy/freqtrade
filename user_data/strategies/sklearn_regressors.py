from typing import Any

import numpy as np
from pandas import DataFrame

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

try:
    from sklearn.linear_model import ElasticNet, LinearRegression
except Exception as e:  # pragma: no cover - import guard
    raise RuntimeError(
        "scikit-learn is required for Sklearn wrappers. Install with `pip install scikit-learn`."
    ) from e


class ElasticNetRegressor(BaseRegressionModel):
    """
    FreqAI wrapper around sklearn.linear_model.ElasticNet.
    Pass ElasticNet keyword arguments via model_training_parameters in config.
    """

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) == 0:
            eval_set = None
            eval_weights = None
        else:
            eval_set = (data_dictionary["test_features"], data_dictionary["test_labels"])  # noqa: F841 (kept for parity)
            eval_weights = data_dictionary["test_weights"]  # noqa: F841

        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"].to_numpy().ravel()
        train_weights = data_dictionary.get("train_weights")

        model = ElasticNet(**self.model_training_parameters)

        if train_weights is not None:
            try:
                model.fit(X=X, y=y, sample_weight=train_weights)
            except TypeError:
                model.fit(X=X, y=y)
        else:
            model.fit(X=X, y=y)

        return model


class LinearRegressionRegressor(BaseRegressionModel):
    """
    FreqAI wrapper around sklearn.linear_model.LinearRegression.
    Re-usable template for other sklearn estimators.
    """

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"].to_numpy().ravel()
        train_weights = data_dictionary.get("train_weights")

        model = LinearRegression(**self.model_training_parameters)

        if train_weights is not None:
            try:
                model.fit(X=X, y=y, sample_weight=train_weights)
            except TypeError:
                model.fit(X=X, y=y)
        else:
            model.fit(X=X, y=y)

        return model
