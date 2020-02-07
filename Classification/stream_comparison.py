from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Utilities.LargeDataProcessing.Sampling import sample_patches
from eolearn.core import FeatureType
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.cluster.hierarchy as sch
from sklearn import tree
from streamdm import HoeffdingTree, HoeffdingAdaptiveTree, NaiveBayes, LogisticRegression, MajorityClass, Perceptron, \
    Bagging
from .classification_comparison import get_data, fit_predict, class_names

methods = [
    #{
    #    'name': 'Decision Tree (scikit-learn)',
    #    'ctor': DecisionTreeClassifier,
    #    'params': {}
    #},
    {
        'name': 'Hoeffding Tree (streamDM)',
        'ctor': HoeffdingTree,
        'params': {
            'max_byte_size': 33554432,
            'memory_estimate_period': 1000000,
            'grace_period': 200,
            'split_confidence': 0.0000001,
            'tie_threshold': 0.05,
            'binary_splits': False,
            'stop_mem_management': False,
            'remove_poor_atts': False,
            'leaf_learner': 'NB',
            'bb_threshold': 0,
            'tree_property_index_list': "",
            'no_pre_prune': False
        }
    },
    {
        'name': 'Hoeffding Adaptive Tree (streamDM)',
        'ctor': HoeffdingAdaptiveTree,
        'params': {
            'max_byte_size': 33554432,
            'memory_estimate_period': 1000000,
            'grace_period': 200,
            'split_confidence': 0.0000001,
            'tie_threshold': 0.05,
            'binary_splits': False,
            'stop_mem_management': False,
            'remove_poor_atts': False,
            'leaf_learner': 'NB',
            'bb_threshold': 0,
            'tree_property_index_list': "",
            'no_pre_prune': False
        }
    },
    {
        'name': 'Bagging (streamDM)',
        'ctor': Bagging,
        'params': {
            'ensemble_size': 10,
            'learner': {
                'name': 'HoeffdingTree',
                'max_byte_size': 33554432,
                'memory_estimate_period': 1000000,
                'grace_period': 200,
                'split_confidence': 0.0000001,
                'tie_threshold': 0.05,
                'binary_splits': False,
                'stop_mem_management': False,
                'remove_poor_atts': False,
                'leaf_learner': 'NB',
                'bb_threshold': 0,
                'tree_property_index_list': "",
                'no_pre_prune': False
            }
        }
    },
    {
        'name': 'Naive Bayes (streamDM)',
        'ctor': NaiveBayes,
        'params': {}
    },
    {
        'name': 'Logistic Regression (streamDM)',
        'ctor': LogisticRegression,
        'params': {
            'learning_ratio': 0.01,
            'lambda': 0.0001
        }
    },
    {
        'name': 'Perceptron (streamDM)',
        'ctor': Perceptron,
        'params': {
            'learning_ratio': 1.0
        }
    },
    {
        'name': 'Majority Class (streamDM)',
        'ctor': MajorityClass,
        'params': {}
    }
]

if __name__ == '__main__':

    # x, y = get_data('../Utilities/LargeDataProcessing/Samples/enriched_samples9797.csv')
    x, y = get_data('../Utilities/LargeDataProcessing/Samples/enriched_samples10000.csv')

    for method in methods:
        learner = method['ctor']()
        learner.set_params(**method['params'])
        fit_predict(x, y, learner, class_names, method['name'])
