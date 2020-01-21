
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Utilities.LargeDataProcessing.Sampling import sample_patches
from eolearn.core import EOTask, EOPatch, LinearWorkflow, FeatureType, OverwritePermission, \
    LoadFromDisk, SaveToDisk, EOExecutor
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Importing the dataset
# path = 'E:/Data/PerceptiveSentinel'
path = '/home/beno/Documents/test/Slovenia'

no_patches = 3
no_samples = 10000
class_feature = (FeatureType.MASK_TIMELESS, 'LPIS_2017')
mask = (FeatureType.MASK_TIMELESS, 'EDGES_INV')
features = [(FeatureType.DATA_TIMELESS, 'ARVI_max_mean_len'),
            (FeatureType.DATA_TIMELESS, 'EVI_min_val'),
            (FeatureType.DATA_TIMELESS, 'NDVI_min_val'),
            (FeatureType.DATA_TIMELESS, 'NDVI_sd_val'),
            (FeatureType.DATA_TIMELESS, 'SAVI_min_val'),
            (FeatureType.DATA_TIMELESS, 'SIPI_mean_val')
            ]
samples_per_class = None
debug = True

feature_names = [t[1] for t in features]

dataset = sample_patches(path, no_patches, no_samples, class_feature, mask, features, samples_per_class, debug)

print('\nClass sample size: {}'.format(int(dataset['LPIS_2017'].size / pd.unique(dataset['LPIS_2017']).size)))
# no_classes = pd.unique(dataset['LPIS_2017']).size

y = dataset['LPIS_2017'].to_numpy()
y = [a + 1 for a in y]

x = dataset[feature_names].to_numpy()
# Splitting the dataset into the Training set and Test set

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
# Feature Scaling

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
d_train = lgb.Dataset(x_train, label=y_train)
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'multiclass'
params['num_class'] = 24
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10
clf = lgb.train(params, d_train, 100)

y_pred = clf.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
# Accuracy

accuracy = accuracy_score(y_pred, y_test)

print(accuracy)
