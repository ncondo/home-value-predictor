import xgboost as xgb
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

from home_value_predictor.models.base import Model


SAVED_MODELS_DIR = Model.model_dirname()/"xgboost"

DEFAULT_PARAMS = {'n_estimators':range(10, 200, 10), 
                  'learning_rate':[0.05,0.060,0.070], 
                  'max_depth':[3,5,7],
                  'min_child_weight':[1,1.5,2]}


class XGBoostModel:
    """Wrapper class for xgboost with convenient functions"""

    def __init__(self, random_state=42):
        self.model = xgb.XGBRegressor(random_state=random_state)
        self.best_params = {}
        self.best_score = None


    def load(self, fname):
        self.model.load_model(SAVED_MODELS_DIR/fname)


    def save(self, fname):
        self.model.save_model(SAVED_MODELS_DIR/fname)


    def train(self, X_train, y_train, params=DEFAULT_PARAMS, save_best=True):
        grid_obj_xgb = RandomizedSearchCV(self.model, 
                                          params,
                                          scoring='r2', 
                                          cv=5,
                                          n_jobs=-1,
                                          n_iter=100,
                                          random_state=99)

        grid_fit_xgb = grid_obj_xgb.fit(X_train, y_train)

        xgb_opt = grid_fit_xgb.best_estimator_
        self.model = xgb_opt
        self.best_params = grid_fit_xgb.best_params_
        self.best_score = grid_fit_xgb.best_score_

        if save_best:
            self.save('xgb_model_'+str(np.around(self.best_score, 4))+'.json')


    def predict(self, X_test, transform_output=False):
        if transform_output:
            preds = self.transform_output(self.model.predict(X_test))
        else:
            preds = self.model.predict(X_test)

        return preds

    
    def transform_output(self, output):
        return np.exp(output) - 1


    def evaluate(self, y_test, preds, metric='r2_score'):
        if metric == 'r2_score':
            return r2_score(y_test, preds)
