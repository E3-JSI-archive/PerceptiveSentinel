# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# from IPython import get_ipython

# %% [markdown]
# # Land Use/Land Cover (LULC) modelling with ml-rapids

# %%
# Setup juptyer notebook
# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')


# %%
# Import all dependencies
import csv
import os
import time

import sklearn
import joblib
import numpy as np
import pandas as pd
from eolearn.core import EOExecutor, EOPatch, EOTask, FeatureType,     LinearWorkflow, LoadTask, SaveTask, OverwritePermission
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix, plot_confusion_matrix,     f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate,     KFold
from tqdm.auto import tqdm

# Machine learning algorithms
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
# from ml_rapids import HoeffdingTree
from lib.streamdm import HoeffdingTree

# %% [markdown]
# ## Dataset preparation

# %%
# Load the sampled dataset from CSV file
data_path = os.path.join(os.getcwd(), 'samples', 'LULC', '2017')
dataset = pd.read_csv(os.path.join(data_path, 'dataset.csv'))
# dataset


# %%
# Load the list of selected feature from CSV file
features = list(pd.read_csv(os.path.join(data_path, 'features.csv')).columns)

print(f'Selected features ({len(features)} / {len(dataset.columns[4:])}):')
for feature in features:
    print(f'{feature}')


# %%
# Prepare input and target values for ML algorithms
X = dataset[features].to_numpy()
y = dataset['LULC'].to_numpy()

labels_unique = np.unique(y)
num_classes = len(labels_unique)

# Normalize input values
# X = StandardScaler().fit_transform(X)

# %% [markdown]
# ## Model evaluation
# 
# We compare methods from 3 different libraries:
# - `LGBMClassifier`: Light Gradinet Boosting Machine from [https://github.com/microsoft/LightGBM](https://github.com/microsoft/LightGBM)
# - `RandomForestClassifier`: Random Forest Classifier from [https://github.com/scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn)
# - `HoeffdingTree`: Hoeffding Tree Classifier from [https://github.com/JozefStefanInstitute/ml-rapids](https://github.com/JozefStefanInstitute/ml-rapids)

# %%
# Configure ML methods
methods = [
    (LGBMClassifier, {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': num_classes,
        'random_state': 42
    }),
    (RandomForestClassifier, {
        'n_estimators': 10,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    }),
    (HoeffdingTree, {
        'max_byte_size': 33554432,
        'memory_estimate_period': 1000000,
        'grace_period': 200,
        'split_confidence': 0.0000001,
        'tie_threshold': 0.5,
        'binary_splits': False,
        'stop_mem_management': False,
        'remove_poor_atts': False,
        'leaf_learner': 'NBAdaptive',
        'nb_threshold': 0,
        'tree_property_index_list': '',
        'no_pre_prune': False
    })
]

# Initialize evaluation report table
report = pd.DataFrame(
    { k: np.zeros(len(methods)) for k in ['Training time', 'Inference time', 'CA', 'F1'] },
    index=[method[0].__name__ for method in methods]
)

# Evaluate with 5-fold cross validation
k = 5
kf = KFold(n_splits=k, random_state=42, shuffle=True)
pbar = tqdm(total=k*len(methods))
for cv, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    for method in methods:
        # Initialize model
        method_name = method[0].__name__
        model = method[0](**method[1])

        # Train model on training set
        training_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - training_time
        report.loc[method_name, 'Training time'] += training_time / k

        # Predict classes on test set
        inference_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - inference_time
        report.loc[method_name, 'Inference time'] += inference_time / k

        # Calulate average classification accuracy
        accuracy = accuracy_score(y_test, y_pred)
        report.loc[method_name, 'CA'] += accuracy / k
        
        # Calculate average F1 score
        f1 = f1_score(y_test, y_pred, average='weighted')
        report.loc[method_name, 'F1'] += f1 / k

        # Update progress bar
        pbar.update()

# Show evaluation report
print(report.round(2))

# %% [markdown]
# ## Model construction
# 
# Next we select two methods and build models on whole dataset.

# %%
# Select methods
selected_methods = [
    # LGBMClassifier,
    RandomForestClassifier,
    HoeffdingTree
]

model_path = os.path.join(os.getcwd(), 'models', 'LULC', '2017')
if not os.path.isdir(model_path):
    os.makedirs(model_path)

for method in tqdm([m for m in methods if m[0] in selected_methods]):
    # Initialize model
    model = method[0](**method[1])
    model_name = method[0].__name__

    # Train the model on whole dataset
    model.fit(X, y)

    # Save the model for later use
    if hasattr(model, 'export_json'):
        # ml-rapids models are exported to JSON
        print(f'Saving model: {model_name}')
        model_path = os.path.join(model_path, f'{model_name}.json')
        model.export_json(model_path)
    else:
        # Other models are pickled
        model_path = os.path.join(model_path, f'{model_name}.pkl')
        joblib.dump(model, model_path)


# %%



