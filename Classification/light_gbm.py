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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Importing the dataset
# path = 'E:/Data/PerceptiveSentinel'
path = '/home/beno/Documents/test/Slovenia'

crop_names = {0: 'Beans', 1: 'Beets', 2: 'Buckwheat', 3: 'Fallow land', 4: 'Grass', 5: 'Hop',
              6: 'Leafy Legumes and/or grass mixture', 7: 'Maize', 8: 'Meadows', 9: 'Orchards', 10: 'Other', 11: 'Peas',
              12: 'Poppy', 13: 'Potatoes', 14: 'Pumpkins', 15: 'Soft fruits', 16: 'Soybean', 17: 'Summer cereals',
              18: 'Sun flower', 19: 'Vegetables', 20: 'Vineyards', 21: 'Winter cereals', 22: 'Winter rape'}

no_patches = 6
no_samples = 10000
features = [(FeatureType.DATA_TIMELESS, 'ARVI_max_mean_len'),
            (FeatureType.DATA_TIMELESS, 'EVI_min_val'),
            (FeatureType.DATA_TIMELESS, 'NDVI_min_val'),
            (FeatureType.DATA_TIMELESS, 'NDVI_sd_val'),
            (FeatureType.DATA_TIMELESS, 'SAVI_min_val'),
            (FeatureType.DATA_TIMELESS, 'SIPI_mean_val')
            ]

feature_names = [t[1] for t in features]

dataset = sample_patches(path=path,
                         no_patches=6,
                         no_samples=10000,
                         class_feature=(FeatureType.MASK_TIMELESS, 'LPIS_2017'),
                         mask_feature=(FeatureType.MASK_TIMELESS, 'EDGES_INV'),
                         features=features,
                         samples_per_class=1000,
                         debug=False,
                         seed=10222)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

# print(dataset)
# print('\nClass sample size: {}'.format(int(dataset['LPIS_2017'].size / pd.unique(dataset['LPIS_2017']).size)))
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

model = lgb.LGBMClassifier(objective='multiclass', num_class=24, metric='multi_logloss')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
# Accuracy
print(cm)

accuracy = accuracy_score(y_pred, y_test)

print('accuracy: ', accuracy)

f1 = f1_score(y_test, y_pred, labels=range(24), average='micro')

print('f1 :', f1)
