import os
import unittest

from home_value_predictor.models.xgboost_model import XGBoostModel
from home_value_predictor.datasets.home_dataset import HomeDataset


class TestXGBoostModel(unittest.TestCase):

    def test_performance(self):
        """Test that the r2_score is at an acceptible level"""

        # load data
        data = HomeDataset()
        processed_df = data.load_data()
        _, X_test, _, y_test = data.split_data(processed_df)

        # load current model
        xgb = XGBoostModel()
        xgb.load('xgb_model.json')

        preds = xgb.predict(X_test)
        r2 = xgb.evaluate(y_test, preds)

        self.assertGreater(r2, 0.85)
