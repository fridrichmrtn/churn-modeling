# Databricks notebook source
#
##
### CUSTOM STEPS

import copy
def reduce_y(func):
    def actual_reduce(self, X, y,  *args, **kwargs):
        if self.y_ is None:
            self.y_ = copy.copy(y)
        if len(y.shape)>1:
            y = y[:,0]
        return func(self, X, y, *args, **kwargs)
    return actual_reduce

def expand_y(func):
    def actual_expand(self, X, y, *args, **kwargs):
        #if len(self.y_.shape)>1:
        y = copy.copy(self.y_)
        return func(self, X, y, *args, **kwargs)
    return actual_expand


# FEATURE SELECTOR
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectorMixin
from imblearn.pipeline import Pipeline
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils.validation import check_is_fitted

class DataFrameTransposer(BaseEstimator, TransformerMixin):
  
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.copy().T

class HierarchicalFeatureSelector(SelectorMixin, BaseEstimator):
  
    def __init__(self, n_features=10, alpha=0.001):
        self.n_features = n_features
        self.alpha = alpha
        self.y_=None    

    def _get_cluster_assignments(self, data):
        data = data.loc[:,self.results_.feature.values]
        n_components = data.shape[1]
        pipe = Pipeline([("rotate", DataFrameTransposer()),
            ("pca", GaussianRandomProjection(n_components=n_components)),
            ("cluster", AgglomerativeClustering(n_clusters=self.n_features))])
        return pipe.fit_predict(data)
    
    @reduce_y
    def _get_correlations(self, X, y):
        tf_corr = [pearsonr(y, X[c]) for c in X.columns]
        correlations = pd.DataFrame(tf_corr, index=X.columns).reset_index()
        correlations.columns = ["feature", "r", "p"]
        correlations["abs_r"] = correlations.r.abs()
        correlations["sf"] = correlations.p<=self.alpha/X.shape[1]
        return correlations
    
    def fit(self, X, y):
        #if len(y.shape)>1:
        #    y = y[:,0]
        X = pd.DataFrame(X)
        #y = pd.Series(y)

        self.in_features_ =  X.columns
        self.results_ = self._get_correlations(X, y)

        if np.sum(self.results_.sf)<= self.n_features:
            self.best_ = self.results_[self.results_.sf]
        else:
            self.results_["cluster"] = self._get_cluster_assignments(X)
            self.best_ = self.results_[self.results_.sf]\
                .merge(self.results_.groupby("cluster",
                    as_index=False).abs_r.max(), on=["cluster", "abs_r"])\
                        .drop_duplicates(["cluster", "abs_r"]).dropna()
        return self
    
    def _get_support_mask(self):
        return np.array([c in set(self.best_.feature) for c in self.in_features_])
    
#
##
### SAMPLING STRATEGIES
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

class _RandomUnderSampler(RandomUnderSampler):
    def __init__(self, sampling_strategy="auto"):
        super().__init__(sampling_strategy=sampling_strategy)
        self.y_=None
    @reduce_y
    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        return self
    @reduce_y
    def fit_resample(self, X, y, **kwargs):
        _ = super().fit_resample(X, y, **kwargs)
        ind = self.sample_indices_
        return (X[ind,:], self.y_[ind])

class _RandomOverSampler(RandomOverSampler):
    def __init__(self, sampling_strategy="auto"):
        super().__init__(sampling_strategy=sampling_strategy)
        self.y_=None
    @reduce_y
    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        return self
    @reduce_y
    def fit_resample(self, X, y, **kwargs):
        _ = super().fit_resample(X, y, **kwargs)
        ind = self.sample_indices_
        return (X[ind,:], self.y_[ind])
      
# MLPC
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from scikeras.wrappers import KerasClassifier

class MLPClassifier(KerasClassifier):

    def __init__(self, n_layers=1, layer_size=8,
        activation="relu", optimizer="adam",
        #optimizer__learning_rate=10**-3,
        epochs=50, verbose=0, **kwargs,):
            super().__init__(**kwargs)
            self.n_layers = n_layers
            self.layer_size = layer_size
            self.activation = activation
            self.optimizer = optimizer
            self.epochs = epochs
            self.verbose = verbose
        
    def _keras_build_fn(self, compile_kwargs):
        model = Sequential()
        inp = Input(shape=(self.n_features_in_))
        model.add(inp)
        for i in range(self.n_layers):
            model.add(Dense(self.layer_size,
                activation=self.activation))
            model.add(BatchNormalization())
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy",
            optimizer=compile_kwargs["optimizer"])
        return model
  

# COMMAND ----------

#
##
### CombiNet
import numpy as np
import pandas as pd
#import copy
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import  _SigmoidCalibration
from sklearn.isotonic import IsotonicRegression

class MultiOutputCalibrationCV(BaseEstimator):
    def __init__(self, base_estimator, method="isotonic", cv=3):
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv

    def fit(self, X, y):
        calibrated_pairs = []
        
        scv = StratifiedKFold(n_splits=2)
        for train_index, test_index in scv.split(X, y[:,0]):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # fit combinet
            base_estimator = clone(self.base_estimator)
            base_estimator.fit(X_train, y_train)
            y_pred = base_estimator.predict_proba(X_test)
        
            # fit calibrator
            if self.method=="isotonic":
                calibrator = IsotonicRegression(y_min=0,y_max=1, out_of_bounds="clip")
                calibrator.fit(y_pred[:,1].T, y_test[:,0])
            if self.method=="sigmoid":
                calibrator = _SigmoidCalibration()
                calibrator.fit(y_pred[:,1].T, y_test[:,0])
            calibrated_pairs.append((base_estimator, calibrator))
        self.calibrated_pairs = calibrated_pairs
        return self

    def predict_proba(self, X):
        # calibrated positive class
        calibrated_class = np.zeros(shape=(X.shape[0], len(self.calibrated_pairs)))
        for i, calibrated_pair in enumerate(self.calibrated_pairs):
            raw_prediction = calibrated_pair[0].predict_proba(X)[:,1]
            calibrated_class[:,i] = calibrated_pair[1].predict(raw_prediction)
        calibrated_class = np.mean(calibrated_class, axis=1)
        return np.column_stack([1-calibrated_class, calibrated_class])

    def predict_reg(self, X):
        calibrated_reg = np.zeros(shape=(X.shape[0], len(self.calibrated_pairs)))
        for i, calibrated_pair in enumerate(self.calibrated_pairs):
            calibrated_reg[:,i] = calibrated_pair[0].predict(X, scope="regression")
        return np.mean(calibrated_reg, axis=1)
    
    def predict_full(self, X):
        return np.column_stack([(self.predict_proba(X)[:,1]>0.5).astype("int"),
            self.predict_reg(X)])
    
    def predict(self, X, scope="classification"):

        if scope=="classification":
            return (self.predict_proba(X)[:,1]>0.5).astype("int")
        if scope=="regression":
            return self.predict_reg(X)
        if scope=="full":
            return self.predict_full(X)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

class MultiOutputTransformer(BaseEstimator, TransformerMixin):

    def fit(self, y):
        if isinstance(y, pd.DataFrame):
            y = y.values
        y_class, y_reg = y[:, 0].reshape(-1,1), y[:, 1].reshape(-1,1)

        self.class_encoder_ = OneHotEncoder(sparse=False)
        self.reg_transformer_ = PowerTransformer()
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
       
from scikeras.wrappers import BaseWrapper
from tensorflow.keras.initializers import HeNormal, LecunNormal, HeNormal
from tensorflow.keras.layers import Input, Dense, BatchNormalization, concatenate, LeakyReLU
from tensorflow.keras import Model

class CombiNet(BaseWrapper):

    def __init__(self, activation = "selu",
        se_layers=1, se_units=256,
        re_layers=5, re_units=100,
        ce_layers=5, ce_units=100, cc_units=75,
        epochs=10, verbose=0,
        optimizer="adam", optimizer__clipvalue=1.0, **kwargs):
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
                kernel_initializer=weight_init)(fe)
            fe = BatchNormalization()(fe)
        # regression branch
        re = fe
        for i in range(self.re_layers):
            re = Dense(self.re_units, self.activation,
                kernel_initializer=weight_init)(re)
            re = BatchNormalization()(re)
        rr_head = Dense(1,"linear")(re)
        # classification branch
        ce = fe
        for i in range(self.ce_layers):
            ce = Dense(self.ce_units, self.activation,
                kernel_initializer=weight_init)(ce)
            ce = BatchNormalization()(ce)
        cc = Dense(self.cc_units, self.activation,
            kernel_initializer=weight_init)(concatenate([ce, re]))
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

# COMMAND ----------

#
##
### SETUP PREPROCESSING

from hyperopt import hp
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.calibration import CalibratedClassifierCV

scaling = [RobustScaler(), QuantileTransformer(), PowerTransformer()]
sampling = [_RandomOverSampler(sampling_strategy="auto"), _RandomUnderSampler(sampling_strategy="auto"),"passthrough"]

preprocessing = {
    "smooth":
        {"steps":
            [("variance_filter", VarianceThreshold()),
            ("data_scaler", PowerTransformer()),
            ("feature_selector", HierarchicalFeatureSelector()),
            ("data_sampler", _RandomUnderSampler(sampling_strategy="auto"))],
        "space":
            {"variance_filter__threshold":hp.uniform("variance_filter__threshold", 10**-1, 5*10**1),
            "data_scaler":hp.choice("data_scaler", scaling),
            "feature_selector__n_features":hp.randint("feature_selector__n_features", 5, 100),
            "data_sampler":hp.choice("data_sampler", sampling)
            }},
    "tree":
         {"steps":
              [("variance_filter", VarianceThreshold()),
              ("data_sampler", "passthrough")
              ],
         "space":
             {"variance_filter__threshold":hp.uniform("variance_filter__threshold", 10**-2, 5*10**1),
             "data_sampler":hp.choice("data_sampler", sampling)
             }}}

calibration = {
    "standard": CalibratedClassifierCV,
    "multi-output": MultiOutputCalibrationCV}

#
##
### SETUP MODELS
from hyperopt import hp
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import Nystroem
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from lightgbm import LGBMClassifier

models = {
    "lr":
          {"model":
               [("lr", LogisticRegression(solver="saga",
                   penalty="elasticnet", max_iter=200))],          
           "space":
               {"lr__C":hp.uniform("lr__C",10**-2,10**1),
                "lr__l1_ratio":hp.uniform("lr__l1_ratio",0,1)},
           "preprocessing":"smooth",
          "type":"standard"},
    
    "svm_lin":
          {"model":
               [("svm_lin", LinearSVC(dual=False))],
           "space":
               {"svm_lin__C":hp.uniform("svm_lin__C",10**-2,10**1),
                "svm_lin__penalty":hp.choice("svm_lin__penalty",["l1","l2"])},
          "preprocessing":"smooth",
          "type":"standard"},
    
    "svm_rbf":
          {"model":
               [("rbf", Nystroem()),
                ("svm_lin", LinearSVC(dual=False))],
           "space":
               {"rbf__n_components":hp.randint("rbf__n_components",20,100),
                "svm_lin__C":hp.uniform("svm_lin__C",10**-2,10**1),
                "svm_lin__penalty":hp.choice("svm_lin__penalty",["l1","l2"])},
          "preprocessing":"smooth",
          "type":"standard"},
    
    "mlp":
          {"model":
               [("mlp", MLPClassifier())],
           "space":
               {"mlp__batch_size":hp.randint("mlp__batch_size",2**3,2**6),
                "mlp__epochs":hp.randint("mlp__epochs",10**2,10**3),
                "mlp__n_layers":hp.randint("mlp__n_layers",1,10),
                "mlp__layer_size":hp.randint("mlp__layer_size",2**2,2**8),
                "mlp__activation":hp.choice("mlp__activation",
                    ["tanh", "sigmoid", "relu", keras.layers.LeakyReLU()]),
                "mlp__optimizer__learning_rate":hp.uniform("mlp__optimizer__learning_rate", 10**-5,10**-3),
                "mlp__optimizer":hp.choice("mlp__optimizer",["sgd", "adam", "rmsprop"])},
          "preprocessing":"smooth",
          "type":"standard"},
    
    "dt":
          {"model":
               [("dt", DecisionTreeClassifier())],
           "space":
               {"dt__max_depth":hp.randint("dt__max_depth",2,30),
                "dt__min_samples_split":hp.randint("dt__min_samples_split",10**1,2*10**2),
                "dt__min_samples_leaf":hp.randint("dt__min_samples_leaf",1,100),
                #"dt__ccp_alpha":hp.uniform("dt__base_estimator__ccp_alpha",0,1),
                "dt__min_impurity_decrease":hp.uniform("dt__min_impurity_decrease",0,.05),
                "dt__min_weight_fraction_leaf":hp.uniform("dt__min_weight_fraction_leaf",0,.05),
               },
          "preprocessing":"tree",
          "type":"standard"},
    
    "rf":
          {"model":
               [("rf", RandomForestClassifier())],
           "space":
               {"rf__n_estimators":hp.randint("rf__n_estimators",25,500),
                "rf__max_features":hp.uniform("rf__max_features",0.1,.7),
                "rf__max_depth":hp.randint("rf__max_depth",2,30),
                "rf__min_samples_split":hp.randint("rf__min_samples_split",10**1,2*10**2),
                "rf__min_samples_leaf":hp.randint("rf__min_samples_leaf",1,100),
                #"dt__ccp_alpha":hp.uniform("dt__ccp_alpha",0,0.05),
                "rf__min_impurity_decrease":hp.uniform("rf__min_impurity_decrease",0,.05),
                "rf__min_weight_fraction_leaf":hp.uniform("rf__min_weight_fraction_leaf",0,.05),
               },
          "preprocessing":"tree",
          "type":"standard"},
    
   "hgb":
         {"model":
              [("hgb", HistGradientBoostingClassifier())],
          "space":
              {
               "hgb__learning_rate":hp.uniform("hgb__learning_rate",0.01,.15),
               "hgb__max_iter":hp.randint("hgb__max_iter",25,500),
               "hgb__max_leaf_nodes":hp.randint("hgb__max_leaf_nodes",5**2,5**3),
               "hgb__max_depth":hp.randint("hgb__max_depth",2,30),
               "hgb__min_samples_leaf":hp.randint("hgb__min_samples_leaf",1,100),
               "hgb__l2_regularization":hp.uniform("hgb__l2_regularization",0,10**2),                
       },
         "preprocessing":"tree",
         "type":"standard"},
    
    "combinet":
         {"model":
              [("combinet", CombiNet())],
           "space":
               {
                "combinet__se_layers":hp.randint("combinet__se_layers",1,5),
                "combinet__se_units":hp.randint("combinet__se_units",2**5,2**9),
                "combinet__re_layers":hp.randint("combinet__re_layers",1,10),
                "combinet__re_units":hp.randint("combinet__re_units",2**5,2**8),                
                "combinet__ce_layers":hp.randint("combinet__ce_layers",1,10),
                "combinet__ce_units":hp.randint("combinet__ce_units",2**5,2**8), 
                "combinet__cc_units":hp.randint("combinet__cc_units",2**5,2**7), 
                "combinet__activation":hp.choice("combinet__activation",
                    ["selu", keras.layers.LeakyReLU()]),
                "combinet__batch_size":hp.randint("combinet__batch_size",2**3,2**6),
                "combinet__epochs":hp.randint("combinet__epochs",10**2,10**3),   
                "combinet__optimizer__learning_rate":hp.uniform("combinet__optimizer__learning_rate", 10**-5,10**-3),
                "combinet__optimizer":hp.choice("mlp__optimizer",["sgd", "adam", "rmsprop"])
               },
         "preprocessing":"smooth",
         "type":"multi-output"},
    
}

#
##
### SETUP PIPELINES
pipelines = {k:{"steps":Pipeline(preprocessing[v["preprocessing"]]["steps"]+v["model"]),
    "space":dict(preprocessing[v["preprocessing"]]["space"],**v["space"]),
    "calibration":calibration[v["type"]],
   "type":v["type"],
    "name":k} for k, v in models.items()}

def get_pipeline(name):
    return pipelines[name]
