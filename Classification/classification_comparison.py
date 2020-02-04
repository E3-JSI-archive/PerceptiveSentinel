from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Utilities.LargeDataProcessing.Sampling import sample_patches
from eolearn.core import FeatureType

import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import time
import pandas as pd

crop_names = {0: 'Beans', 1: 'Beets', 2: 'Buckwheat', 3: 'Fallow land', 4: 'Grass', 5: 'Hop',
              6: 'Leafy Legumes and/or grass mixture', 7: 'Maize', 8: 'Meadows', 9: 'Orchards', 10: 'Other',
              11: 'Peas',
              12: 'Poppy', 13: 'Potatoes', 14: 'Pumpkins', 15: 'Soft fruits', 16: 'Soybean', 17: 'Summer cereals',
              18: 'Sun flower', 19: 'Vegetables', 20: 'Vineyards', 21: 'Winter cereals', 22: 'Winter rape'}

features = [(FeatureType.DATA_TIMELESS, 'ARVI_max_mean_len'),
            (FeatureType.DATA_TIMELESS, 'EVI_min_val'),
            (FeatureType.DATA_TIMELESS, 'NDVI_min_val'),
            (FeatureType.DATA_TIMELESS, 'NDVI_sd_val'),
            (FeatureType.DATA_TIMELESS, 'SAVI_min_val'),
            (FeatureType.DATA_TIMELESS, 'SIPI_mean_val')
            ]


def get_data(path):
    dataset = sample_patches(path=path,
                             no_patches=6,
                             no_samples=10000,
                             class_feature=(FeatureType.MASK_TIMELESS, 'LPIS_2017'),
                             mask_feature=(FeatureType.MASK_TIMELESS, 'EDGES_INV'),
                             features=features,
                             samples_per_class=1000,
                             debug=False,
                             seed=10222)
    return dataset


def fit_predict(x, y, model, name):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    start_time = time.time()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    total_time = time.time() - start_time

    # print(confusion_matrix(y_test, y_pred))
    accuracy = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_test, y_pred, labels=range(24), average='macro')

    stats = '{0:20} CA: {1:.4} F1: {2:.4} time: {3:.3}'.format(name, accuracy, f1, total_time)
    print(stats)


if __name__ == '__main__':
    # path = 'E:/Data/PerceptiveSentinel'
    # path = '/home/beno/Documents/test/Slovenia'
    # dataset = get_data(path)

    samples_path = 'D:\Samples\enriched_samples9797.csv'
    dataset = pd.read_csv(samples_path)
    # dataset.dropna(axis=0, inplace=True)
    dataset.drop(columns=['INCLINATION'])
    # dataset.drop(columns=['NDVI_min_val', 'SAVI_min_val', 'INCLINATION'])
    y = dataset['LPIS_2017'].to_numpy()
    # !!!! -1 is marking no LPIS data so everything is shifted by one cause some classifiers don't want negative numbers
    y = [a + 1 for a in y]

    feature_names = [t[1] for t in features]
    x = dataset[feature_names].to_numpy()

    lgb_model = lgb.LGBMClassifier(objective='multiclass', num_class=24, metric='multi_logloss')
    fit_predict(x, y, lgb_model, 'LightGBM')

    '''
    rf_model = RandomForestClassifier()
    fit_predict(x, y, lgb_model, 'Random forest')

    lr_model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200)
    fit_predict(x, y, lr_model, 'Logistic Regression')
    '''