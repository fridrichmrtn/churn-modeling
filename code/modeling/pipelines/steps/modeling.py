# Databricks notebook source
import numpy as np
import pandas as pd
import os
from tensorflow.keras import Sequential, Model
from tensorflow.keras.initializers import HeNormal, LecunNormal, HeNormal
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, concatenate

from scikeras.wrappers import KerasClassifier, KerasRegressor, BaseWrapper
from sklearn.base import BaseEstimator,TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, QuantileTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet
from sklearn.calibration import CalibratedClassifierCV, _SigmoidCalibration
from sklearn.ensemble import BaggingRegressor

# COMMAND ----------

#
##
### MULTILAYER PERCEPTRON

class MLPClassifier(KerasClassifier):

    def __init__(self, layers=1, units=8,
        activation="relu", optimizer="adam",
        optimizer__clipvalue=1.0,
        #optimizer__learning_rate=10**-3,
        epochs=50, verbose=0, **kwargs,):
            super().__init__(**kwargs)
            self.layers = layers
            self.units = units
            self.activation = activation
            self.optimizer = optimizer
            self.epochs = epochs
            self.verbose = verbose
            self.optimizer__clipvalue=optimizer__clipvalue
            
    def _get_weight_init(self):
        if isinstance(self.activation, LeakyReLU): 
            init = HeNormal()
        elif self.activation in ["selu", "elu"]:
            init = LecunNormal()
        else:
            init = HeNormal()  
        return init            
        
    def _keras_build_fn(self, compile_kwargs):
        weight_init = self._get_weight_init()
        model = Sequential()
        inp = Input(shape=(self.n_features_in_))
        model.add(inp)
        for i in range(self.layers):
            model.add(Dense(self.units,
                activation=self.activation,
                kernel_initializer=weight_init))
            model.add(BatchNormalization())
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy",
            optimizer=compile_kwargs["optimizer"])
        return model
      
class MLPRegressor(KerasRegressor):

    def __init__(self, layers=1, units=8,
        activation="relu", optimizer="adam",
        optimizer__clipvalue=1.0,
        epochs=50, verbose=0, **kwargs,):
            super().__init__(**kwargs)
            self.layers = layers
            self.units = units
            self.activation = activation
            self.optimizer = optimizer
            self.epochs = epochs
            self.verbose = verbose
            self.optimizer__clipvalue = optimizer__clipvalue
    
    def _get_weight_init(self):
        if isinstance(self.activation, LeakyReLU): 
            init = HeNormal()
        elif self.activation in ["selu", "elu"]:
            init = LecunNormal()
        else:
            init = HeNormal()  
        return init
    
    def _keras_build_fn(self, compile_kwargs):
        weight_init = self._get_weight_init()
        model = Sequential()
        inp = Input(shape=(self.n_features_in_))
        model.add(inp)
        for i in range(self.layers):
            model.add(Dense(self.units,
                activation=self.activation,
                kernel_initializer=weight_init))
            model.add(BatchNormalization())
        model.add(Dense(1, activation="linear"))
        model.compile(loss="mean_squared_error",
            optimizer=compile_kwargs["optimizer"])
        return model        

# COMMAND ----------

#
##
### ELASTICNET

class ElasticNetC(ElasticNet):
    def __init__(self, C=1.0, l1_ratio=0.5, **kwargs):
        super().__init__(alpha = C, l1_ratio=l1_ratio, **kwargs)
        self.C = C

# COMMAND ----------

#
##
### CALIBRATION

class CalibratedClassifierCV(CalibratedClassifierCV):
    def __init__(self, base_estimator, n_jobs=4, **kwargs):
        super().__init__(base_estimator, **kwargs)
        self.n_jobs = n_jobs
        
    def predict(self, X):
        return self.predict_proba(X)[:,1]

class CalibratedPassthrough(BaseEstimator):
    def __init__(self, base_estimator):
        self.base_estimator = clone(base_estimator)
        self.fitted = None
        
    def fit(self, X, y):
        self.fitted = self.base_estimator.fit(X,y)
        return self
    
    def predict(self, X):
        return self.fitted.predict(X)
      
class CalibratedRegression(BaggingRegressor):
    def __init__(self, base_estimator, max_samples=0.8,
        n_jobs=4, **kwargs):
        super().__init__(base_estimator, **kwargs)
        self.max_samples = max_samples
        self.n_jobs = n_jobs

# COMMAND ----------

#
##
### COMBINET
# NOTE: remove this in the future

class MultiOutputCalibrationCV(BaseEstimator):
    
    def __init__(self, base_estimator, method="sigmoid", cv=3):
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv
        self.threshold = .5
        
    # threshold optimization
    def _get_threshold(self, y,prob):
        from sklearn.metrics import roc_curve
        tpr, fpr, thresholds = roc_curve(y, prob)
        scores = 2*tpr*(1-fpr)/(1+tpr-fpr+10**-5)
        return thresholds[np.argmax(scores)]        

    def fit(self, X, y):
        calibrated_pairs = []
        thresholds = []
        scv = StratifiedKFold(n_splits=self.cv)
        for train_index, test_index in scv.split(X, y[:,0]):
            X_train, y_train  = X[train_index], y[train_index] 
            X_test, y_test = X[test_index], y[test_index]

            # fit combinet
            base_estimator = clone(self.base_estimator)
            base_estimator.fit(X_train, y_train)
            y_pred = base_estimator.predict_proba(X_test)
        
            # fit calibrator
            if self.method=="isotonic":
                calibrator = IsotonicRegression(out_of_bounds="clip")
            if self.method=="sigmoid":
                calibrator = _SigmoidCalibration()
            calibrator.fit(y_pred[:,1].T, y_test[:,0])
            calibrated_pairs.append((base_estimator, calibrator)) 
            thresholds.append(self._get_threshold(y_train[:,0],
                calibrator.predict(base_estimator.predict_proba(X_train)[:,1])))
        self.threshold = np.median(thresholds)
        self.calibrated_pairs = calibrated_pairs
        return self

    def predict_proba(self, X):
        # calibrated positive class
        calibrated_class = np.zeros(shape=(X.shape[0], len(self.calibrated_pairs)))
        for i, calibrated_pair in enumerate(self.calibrated_pairs):
            raw_prediction = calibrated_pair[0].predict_proba(X)[:,1]
            calibrated_class[:,i] = raw_prediction
            #calibrated_class[:,i] = calibrated_pair[1].predict(raw_prediction.T)
        calibrated_class = np.mean(calibrated_class, axis=1)
        return np.column_stack([1-calibrated_class, calibrated_class])

    def predict_reg(self, X):
        calibrated_reg = np.zeros(shape=(X.shape[0], len(self.calibrated_pairs)))
        for i, calibrated_pair in enumerate(self.calibrated_pairs):
            calibrated_reg[:,i] = calibrated_pair[0].predict(X, scope="regression")
        return np.mean(calibrated_reg, axis=1)
    
    def predict_full(self, X):
        return np.column_stack([(self.predict_proba(X)[:,1]>self.threshold).astype("int"),
            self.predict_reg(X)])
    
    def predict(self, X, scope="classification"):
        if scope=="classification":
            return (self.predict_proba(X)[:,1]>self.threshold).astype("int")
        if scope=="regression":
            return self.predict_reg(X)
        if scope=="full":
            return self.predict_full(X)

class MultiOutputTransformer(BaseEstimator, TransformerMixin):

    def fit(self, y):
        if isinstance(y, pd.DataFrame):
            y = y.values
        y_class, y_reg = y[:, 0].reshape(-1,1), y[:, 1].reshape(-1,1)

        self.class_encoder_ = OneHotEncoder(sparse=False)
        self.reg_transformer_ = QuantileTransformer()
        # Fit them to the input data
        self.class_encoder_.fit(y_class)
        self.reg_transformer_.fit(y_reg)
        # Save the number of classes
        self.n_classes_ = len(self.class_encoder_.categories_)
        self.n_outputs_expected_ = 2
        return self

    def transform(self, y):
        if isinstance(y, pd.DataFrame):
            y = y.values
        y_class, y_reg = y[:, 0].reshape(-1,1), y[:, 1].reshape(-1,1)
        # Apply transformers to input array
        y_class = self.class_encoder_.transform(y_class)
        y_reg = self.reg_transformer_.transform(y_reg)
        # Split the data into a list
        return [y_class, y_reg]

    def inverse_transform(self, y, return_proba=False):
        y_pred_reg = y[1]
        if return_proba:
            return y[0]
        else:
            y_pred_class = np.zeros_like(y[0])
            y_pred_class[np.arange(len(y[0])), np.argmax(y[0], axis=1)] = 1
            y_pred_class = self.class_encoder_.inverse_transform(y_pred_class)
        y_pred_reg = self.reg_transformer_.inverse_transform(y_pred_reg)
        return np.column_stack([y_pred_class, y_pred_reg])

    def get_metadata(self):
        return {
            "n_classes_": self.n_classes_,
            "n_outputs_expected_": self.n_outputs_expected_}

class CombiNet(BaseWrapper):

    def __init__(self, activation = LeakyReLU(),
        se_layers=1, se_units=256,
        re_layers=5, re_units=100,
        ce_layers=5, ce_units=100, cc_units=50,
        epochs=250, verbose=0,
        optimizer="rmsprop", optimizer__clipvalue=1.0, **kwargs):
            super().__init__(**kwargs)
            self.activation = activation
            self.se_layers = se_layers
            self.se_units = se_units
            self.re_layers = re_layers
            self.re_units = re_units
            self.ce_layers = ce_layers
            self.ce_units = ce_units
            self.cc_units = cc_units
            self.epochs = epochs
            self.verbose = verbose
            self.optimizer = optimizer
            self.optimizer__clipvalue = optimizer__clipvalue
            self.__prediction_scope = {"classification":0,"regression":1,"full":range(2)}

    def _get_weight_init(self):
        if isinstance(self.activation, LeakyReLU): 
            init = HeNormal()
        elif self.activation in ["selu", "elu"]:
            init = LecunNormal()
        else:
            init = HeNormal()  
        return init

    def _keras_build_fn(self, compile_kwargs):
        weight_init = self._get_weight_init()

        # shared extraction
        inp = Input(shape=(self.n_features_in_))
        fe = inp
        for i in range(self.se_layers):
            fe = Dense(self.se_units, self.activation,
                kernel_initializer=weight_init,
                kernel_regularizer="l2")(fe)
            fe = BatchNormalization()(fe)
        # regression branch
        re = fe
        for i in range(self.re_layers):
            re = Dense(self.re_units, self.activation,
                kernel_initializer=weight_init,
                kernel_regularizer="l2")(re)
            re = BatchNormalization()(re)
        rr_head = Dense(1,"linear")(re)
        # classification branch
        ce = fe
        for i in range(self.ce_layers):
            ce = Dense(self.ce_units, self.activation,
                kernel_initializer=weight_init,
                kernel_regularizer="l2")(ce)
            ce = BatchNormalization()(ce)
        cc = Dense(self.cc_units, self.activation,
            kernel_initializer=weight_init,
            kernel_regularizer="l2")(concatenate([ce, re]))
        cc = BatchNormalization()(cc)
        cc_head = Dense(2, "softmax")(cc)

        model = Model(inputs=inp, outputs=[cc_head, rr_head])
        model.compile(loss=["categorical_crossentropy","mse"], loss_weights=[.5,.5],
            optimizer=compile_kwargs["optimizer"])
        return model
        
    @property
    def target_encoder(self):
        return MultiOutputTransformer()
        
    def predict_proba(self, X):
        X = self.feature_encoder_.transform(X)
        y_pred = self.model_.predict(X)
        return self.target_encoder_.inverse_transform(y_pred, return_proba=True)

    def predict(self, X, scope="classification"):
        X = self.feature_encoder_.transform(X)
        y_pred = self.model_.predict(X)
        y_pred = self.target_encoder_.inverse_transform(y_pred)
        return y_pred[:,self.__prediction_scope [scope]]
