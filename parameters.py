"""
#####################################################
from: https://github.com/BioSystemsUM/propythia
####################################################
"""

from scipy.stats import randint, loguniform

def param_shallow():
    param = {'svm':
                 {'param_grid':
                      [{'clf__C': [0.01, 0.1, 1.0, 10],
                        'clf__kernel': ['linear'],
                        'selector__k': [578, 320, 298, 120]},
                       {'clf__C': [0.01, 0.1, 1.0, 10],
                        'clf__kernel': ['rbf'],
                        'clf__gamma': ['scale', 0.001, 0.0001],
                        'selector__k': [578, 320, 298, 120]}],
                  'distribution':
                      [{'clf__C': loguniform(1e-2, 1e1),
                        'clf__kernel': ['linear'],
                        'selector__k': [578, 320, 298, 120]},
                       {'clf__C': loguniform(1e-2, 1e1),
                        'clf__gamma': loguniform(1e-4, 1e1),  # np.power(10, np.arange(-4, 1, dtype=float)),
                        'selector__k': [578, 320, 298, 120]
                        }]
                  },

             'rf':
                 {'param_grid':
                      [{'clf__n_estimators': [10, 100, 500],
                        'clf__max_features': ['sqrt', 'log2'],
                        'clf__bootstrap': [True],
                        'clf__criterion': ["gini"],
                        'selector__k': [578, 320, 298, 120]}],
                  'distribution':
                      [{'clf__n_estimators': randint(1e1, 1e3),
                        'clf__max_features': ['sqrt', 'log2'],
                        'clf__bootstrap': [True],
                        'clf__criterion': ["gini"],
                        'selector__k': [578, 320, 298, 120]}]
                  },

             'gboosting':
                 {'param_grid':
                      [{'clf__n_estimators': [10, 100, 500],
                        'clf__max_depth': [1, 3, 5, 10],
                        'clf__max_features': [0.6, 0.9],
                        'selector__k': [578, 320, 298, 120]}],
                  'distribution':
                      [{'clf__learning_rate': loguniform(1e-3, 1e0),
                        'clf__n_estimators': randint(1e1, 1e3),
                        'clf__max_depth': randint(1e0, 2e1),
                        'clf__max_features': loguniform(5e-1, 1),
                        'selector__k': [578, 320, 298, 120]}]
                  },

             'xgboost':
                 {'param_grid':
                      [{'clf__n_estimators': [10, 100, 500],
                        'clf__max_depth': [1, 3, 5, 10],
                        'selector__k': [578, 320, 298, 120]}],
                  'distribution':
                      [{'clf__learning_rate': loguniform(1e-3, 1e0),
                        'clf__n_estimators': randint(1e1, 1e3),
                        'clf__max_depth': randint(1e0, 2e1),
                        'selector__k': [578, 320, 298, 120]}]
                  },

             'lr':
                 {'param_grid':
                      [{'clf__C': [0.01, 0.1, 1.0, 10.0],
                        'clf__solver': ['liblinear', 'lbfgs', 'sag'],
                        'selector__k': [578, 320, 298, 120]}],
                  'distribution':
                      [{'clf__C': loguniform(1e-2, 1e1),
                        'clf__solver': ['liblinear', 'lbfgs', 'sag'],
                        'selector__k': [578, 320, 298, 120]}]
                  },

             'nn':
                 {'param_grid':
                      [{'clf__activation': ['logistic', 'tanh', 'relu'],
                        'clf__alpha': [0.00001, 0.0001, 0.001],
                        'clf__learning_rate_init': [0.0001, 0.001, 0.01],
                        'clf__early_stopping': [True],
                        'clf__validation_fraction': [0.2],
                        'clf__n_iter_no_change': [50],
                        'selector__k': [578, 320, 298, 120]}],
                  'distribution':
                      [{'clf__activation': ['logistic', 'tanh', 'relu'],
                        'clf__alpha': loguniform(1e-5, 1e-3),
                        'clf__learning_rate_init': loguniform(1e-4, 1e-2),
                        'clf__early_stopping': [True],
                        'clf__validation_fraction': [0.2],
                        'clf__n_iter_no_change': [50],
                        'selector__k': [578, 320, 298, 120]}]  # put a quarter of the iterations by default
                  },

             }
    return param