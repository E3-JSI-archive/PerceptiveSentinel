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

crop_names = {0: 'Beans', 1: 'Beets', 2: 'Buckwheat', 3: 'Fallow land', 4: 'Grass', 5: 'Hop',
              6: 'Legumes or grass', 7: 'Maize', 8: 'Meadows', 9: 'Orchards', 10: 'Other',
              11: 'Peas',
              12: 'Poppy', 13: 'Potatoes', 14: 'Pumpkins', 15: 'Soft fruits', 16: 'Soybean', 17: 'Summer cereals',
              18: 'Sun flower', 19: 'Vegetables', 20: 'Vineyards', 21: 'Winter cereals', 22: 'Winter rape'}
class_names = ['Not Farmland'] + [crop_names[x] for x in range(23)]

features = [(FeatureType.DATA_TIMELESS, 'ARVI_max_mean_len'),
            (FeatureType.DATA_TIMELESS, 'EVI_min_val'),
            (FeatureType.DATA_TIMELESS, 'NDVI_min_val'),
            (FeatureType.DATA_TIMELESS, 'NDVI_sd_val'),
            (FeatureType.DATA_TIMELESS, 'SAVI_min_val'),
            (FeatureType.DATA_TIMELESS, 'SIPI_mean_val')
            ]


def get_data(samples_path):
    samples_path = '../Utilities/LargeDataProcessing/Samples/enriched_samples9797.csv'
    dataset = pd.read_csv(samples_path)
    # dataset.drop(columns=['INCLINATION'])
    # dataset.drop(columns=['NDVI_min_val', 'SAVI_min_val', 'INCLINATION'])
    y = dataset['LPIS_2017'].to_numpy()
    # !!!! -1 is marking no LPIS data so everything is shifted by one cause some classifiers don't want negative numbers
    y = [a + 1 for a in y]

    feature_names = [t[1] for t in features]
    x = dataset[feature_names].to_numpy()

    # dataset = sample_patches(path=path,
    #                          no_patches=6,
    #                          no_samples=10000,
    #                          class_feature=(FeatureType.MASK_TIMELESS, 'LPIS_2017'),
    #                          mask_feature=(FeatureType.MASK_TIMELESS, 'EDGES_INV'),
    #                          features=features,
    #                          samples_per_class=1000,
    #                          debug=False,
    #                          seed=10222)
    return x, y


def save_figure(plt, file_name):
    plt.savefig(f'Results/{file_name}', dpi=300, bbox_inches='tight')


def cluster_df(df, k=0.5):
    X = df.corr().values
    d = sch.distance.pdist(X)
    L = sch.linkage(d, method='complete')
    ind = sch.fcluster(L, k * d.max(), 'distance')
    columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
    df = df.reindex(columns, axis=1)
    return df, ind


def create_dict(ind, group_names):
    new_dict = dict()
    for ni in range(max(ind) + 1):
        new_name = ''
        for i, names in enumerate(group_names):
            if ni == ind[i]:
                new_name += names + ', '
        new_dict[ni] = new_name[0:-2]
    # print(new_dict)
    no_classes_new = len(new_dict)
    class_names_new = [new_dict[x] for x in range(no_classes_new)]
    return new_dict, class_names_new


def form_clusters(y_test, y_pred, k=0.6):
    confusion = confusion_matrix(y_test, y_pred, normalize='pred')
    ds = pd.DataFrame(confusion, columns=class_names)
    dsc, ind = cluster_df(ds, k)
    ind = [x - 1 for x in ind]
    return ind


def fit_predict(x, y, model, labels, name):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    start_time = time.time()
    model.fit(x_train, y_train)
    total_time = time.time() - start_time
    predict_time = time.time()
    y_pred = model.predict(x_test)
    test_time = time.time() - predict_time

    no_classes = range(len(labels))
    fig, ax = plt.subplots()
    ax.set_ylim(bottom=0.14, top=0)
    plot_confusion_matrix(model, x_test, y_test, labels=no_classes,
                          display_labels=labels,
                          cmap='viridis',
                          include_values=False,
                          xticks_rotation='vertical',
                          normalize='pred',
                          ax=ax)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, labels=no_classes, average='macro')
    stats = '{0:_<40} CA: {1:.4} F1: {2:.4} Train: {3:3.1f}s Predict: {4:3.1f}s'.format(name, accuracy, f1,
                                                                                      total_time,
                                                                                      test_time)
    ax.set_title(stats)
    print(stats)

    save_figure(plt, name + '.png')
    return y_pred, y_test


if __name__ == '__main__':
    x, y = get_data('/home/beno/Documents/IJS/Perceptive-Sentinel/enriched_samples9797.csv')

    # LightGBM
    lgb_model = lgb.LGBMClassifier(objective='multiclassova', num_class=len(class_names), metric='multi_logloss', )
    y_pred, y_test = fit_predict(x, y, lgb_model, class_names, 'LGBM')
    new_index = form_clusters(y_pred, y_test, k=0.5)
    new_dict, class_names_new = create_dict(new_index, class_names)
    clustered_y = [new_index[int(i)] for i in y]
    lgb_model = lgb.LGBMClassifier(objective='multiclassova', num_class=len(class_names_new), metric='multi_logloss')
    fit_predict(x, clustered_y, lgb_model, class_names_new, 'LGBM clustered')

    # DecisionTree
    clf = tree.DecisionTreeClassifier()
    y_pred, y_test = fit_predict(x, y, clf, class_names, 'decision tree')
    new_index = form_clusters(y_pred, y_test, k=0.6)
    _, class_names_new = create_dict(new_index, class_names)
    clustered_y = [new_index[int(i)] for i in y]
    clf = tree.DecisionTreeClassifier()
    fit_predict(x, clustered_y, clf, class_names_new, 'clustered tree')

    # Random Forest
    rf_model = RandomForestClassifier()
    y_pred, y_test = fit_predict(x, y, rf_model, class_names, 'random forest')
    new_index = form_clusters(y_pred, y_test, k=0.6)
    _, class_names_new = create_dict(new_index, class_names)
    clustered_y = [new_index[int(i)] for i in y]
    fit_predict(x, clustered_y, rf_model, class_names_new, 'clustered RF')

    # Logistic Regression
    lr_model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200)
    y_pred, y_test = fit_predict(x, y, lr_model, class_names, 'logistic regression')

    new_index = form_clusters(y_pred, y_test, k=0.6)
    _, class_names_new = create_dict(new_index, class_names)
    clustered_y = [new_index[int(i)] for i in y]
    fit_predict(x, clustered_y, lr_model, class_names_new, 'clustered logistic regression')
