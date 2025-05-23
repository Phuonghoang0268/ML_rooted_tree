import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# For CRF
from sklearn_crfsuite import CRF
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# For deep learning models
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Embedding, Bidirectional
from tensorflow_addons.layers import CRF as TF_CRF
from tensorflow_addons.text import crf_log_likelihood

feature_cols = [ 'closeness_pagerank_ratio',
       'eigen_betweenness_product', 'degree_closeness_ratio',
       'pagerank_katz_ratio', 'log_betweenness', 'log_pagerank',
       'high_closeness', 'high_betweenness', 'high_eigenvector', 'closeness_z',
       'pagerank_z', 'betweenness_z', 'eigenvector_z', 'degree_z', 'katz_z',
       'closeness_pagerank_z_sum', 'eigen_betweenness_z_product',
       'avg_rank', 'clustering_coef', 'eccentricity',
       'num_leaf_neighbors']


# feature_cols = ['degree_centrality', 'in_degree']
#
# cols_to_scale = [
#         'degree_centrality',
#     ]



model_class = {
    # 'random forest': {
    #     'model_class': RandomForestClassifier,
    #     'param_grid': {
    #         'n_estimators': [100, 200],
    #         'max_depth': [None, 10, 20],
    #         'min_samples_split': [2, 5],
    #     }
    # },
    # 'lgbm': {
    #     'model_class': LGBMClassifier,
    #     'param_grid': {
    #         'num_leaves': [31, 50],
    #         'learning_rate': [0.1, 0.01],
    #         'n_estimators': [100, 200]
    #     }
    # },
    # 'svc': {
    #     'model_class': SVC,
    #     'param_grid': {
    #         'class_weight':['balanced'],
    #         'C': [0.1, 1, 10],
    #         'kernel': ['linear', 'rbf'],
    #         'gamma': ['scale', 'auto']
    #     }
    # },
    # 'balanced bagging': {
    #     'model_class': BalancedBaggingClassifier,
    #     'param_grid': {
    #         'n_estimators': [10, 50, 100],  # Number of base estimators
    #         'max_samples': [0.5, 0.7, 1.0],  # Fraction of samples to draw for each base estimator
    #         'max_features': [0.5, 0.7, 1.0],  # Fraction of features to draw for each base estimator
    #         'bootstrap': [True, False],  # Whether samples are drawn with replacement
    #         'bootstrap_features': [True, False],  # Whether features are drawn with replacement
    #         'sampling_strategy': ['auto', 0.5, 1.0]  # Sampling strategy for balancing
    #     }
    # },
    # 'xgb': {
    #     'model_class': XGBClassifier,
    #     'param_grid': {
    #         'max_depth': [1,3, 6],
    #         'learning_rate': [0.1, 0.01],
    #         'n_estimators': [100, 200]
    #     }
    # },
    'decision tree': {
        'model_class': DecisionTreeClassifier,
        'param_grid': {
            'class_weight':['balanced'],
            'random_state': [42],
            'max_depth': [None,1, 10, 20]
        }
    },
    'qda': {
        'model_class': QuadraticDiscriminantAnalysis,
        'param_grid': {
            'reg_param': [0.0, 0.1, 1.0],
            'tol': [1e-4],
            'store_covariance': [True]
        }
    },
    # 'logistic regression': {
    #     'model_class': LogisticRegression,
    #     'param_grid': {
    #         'C': [0.1, 1, 10],
    #         'penalty': ['l2'],
    #         'solver': ['lbfgs']
    #     }
    # },
    # 'knn': {
    #     'model_class': KNeighborsClassifier,
    #     'param_grid': {
    #         'n_neighbors': [1, 3, 5, 7],
    #         'weights': ['uniform', 'distance']
    #     }
    # },
    'adaboost': {
        'model_class': AdaBoostClassifier,
        'param_grid': {
            'n_estimators': [10,50, 100],
            'learning_rate': [1.0, 0.1]
        }
    },
    # 'gradient boosting': {
    #     'model_class': GradientBoostingClassifier,
    #     'param_grid': {
    #         'n_estimators': [100, 200],
    #         'learning_rate': [0.1, 0.01],
    #         'max_depth': [3, 5]
    #     }
    # },
    # 'mlp': {
    #     'model_class': MLPClassifier,
    #     'param_grid': {
    #         'hidden_layer_sizes': [(100,), (50, 50)],
    #         'activation': ['relu', 'tanh'],
    #         'solver': ['adam']
    #     }
    # },
    # 'bagging': {
    #     'model_class': BaggingClassifier,
    #     'param_grid': {
    #         'n_estimators': [10, 50],
    #         'max_samples': [0.5, 1.0]
    #     }
    # },
    # 'lda': {
    #     'model_class': LinearDiscriminantAnalysis,
    #     'param_grid': {
    #         'solver': ['svd', 'lsqr', 'eigen']
    #     }
    # }
}


# Create a list of models with their names
models = [
    # ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    # ('LightGBM', LGBMClassifier(class_weight='balanced')),
    # ('SVM', SVC(kernel='rbf', class_weight='balanced', probability=True)),
    # ('Balanced Bagging', BalancedBaggingClassifier(
    #     estimator=DecisionTreeClassifier(),
    #     sampling_strategy='auto',
    #     replacement=False,
    #     random_state=42)),
    # ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=1)),
    # ('Decision Tree', DecisionTreeClassifier(class_weight='balanced', random_state=42, max_depth=1)),
    # ('QDA', QuadraticDiscriminantAnalysis()),
    # ('Logistic Regression', LogisticRegression(class_weight='balanced', solver='liblinear')),
    # ('KNN', KNeighborsClassifier(n_neighbors=5)),
    # ('AdaBoost', AdaBoostClassifier(n_estimators=100, random_state=42)),
    # ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    # ('MLP Classifier', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)),
    ('Bagging Classifier', BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=50,
        random_state=42)),
    ('LDA', LinearDiscriminantAnalysis()),

]
