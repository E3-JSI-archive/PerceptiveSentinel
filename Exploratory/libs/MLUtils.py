import bisect

import numpy as np
from sklearn.model_selection import train_test_split

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def split_train_test_cv(data, train_p=0.5, cv_p=0.5, shuffle=None, state=None):
    if shuffle is 0:
        print("Probably not what you want")
    train, test = train_test_split(data, train_size=train_p, shuffle=shuffle,
                                   random_state=state)
    cv, test = train_test_split(test, train_size=cv_p, shuffle=shuffle,
                                random_state=state)
    return train, cv, test


def split_train_test_data(X, Y, train_p=0.5, cv_p=0.5, shuffle=None, state=None,
                          **kwargs):
    if shuffle is 0:
        print("Probably not what you want")
    train, test, train_labels, test_labels = train_test_split(X, Y,
                                                              train_size=train_p,
                                                              shuffle=shuffle,
                                                              random_state=state)
    cv, test, cv_labels, test_labels = train_test_split(test, test_labels,
                                                        train_size=cv_p,
                                                        shuffle=shuffle,
                                                        random_state=state)
    return (train, cv, test), (train_labels, cv_labels, test_labels)


class Classifiers:
    DECISION_TREE = tree.DecisionTreeClassifier
    RANDOM_FOREST = RandomForestClassifier
    LOGISTIC_REGRESSION = LogisticRegression
    SVC = svm.SVC

    ALL_NAMED = {
        "Decision Tree": DECISION_TREE,
        "Random forest": RANDOM_FOREST,
        "Logistic regression": LOGISTIC_REGRESSION,
    }

    ALL = [j for (_, j) in ALL_NAMED.items()]


class Predictor:
    def __init__(self, model, flatten_data, revert_data, flatten_labels,
                 revert_labels, test_data, test_labels, normalizator):
        self.model = model
        self.flatten_data = flatten_data
        self.revert_data = revert_data
        self.flatten_labels = flatten_labels
        self.revert_labels = revert_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.normalizator = normalizator

    def predict(self, data, reshape=True):
        data = self.normalizator(data)
        flat, dimen = self.flatten_data(data)
        predicted = self.model.predict(flat)
        if reshape:
            return self.revert_labels(predicted, dimen[0])
        return predicted

    def test_classifier(self, X, Y):
        test_labels = self.flatten_labels(Y)
        predicted_labels = self.predict(X, reshape=False)
        # evaluate results
        precission = precision_score(test_labels, predicted_labels)
        recall = recall_score(test_labels, predicted_labels)
        f1 = f1_score(test_labels, predicted_labels)

        return (precission, recall, f1), self.revert_labels(predicted_labels,
                                                            X.shape)


def train_test_classifier(X, Y,
                          classifier_method=Classifiers.RANDOM_FOREST,
                          **config):
    # Reshape
    initial_data_shape = X.shape
    initial_label_shape = Y.shape

    full_size = 1
    for j in X.shape[:-1]:
        full_size *= j

    def revert_data_shape(data):
        return data.reshape(initial_data_shape)

    def revert_labels_shape(labels, shape):
        return labels.reshape(shape[:-1])

    def flatten_data(data):
        initial_shape = data.shape
        full_sz = 1
        for j in data.shape[:-1]:
            full_sz *= j
        # Sounds good, but we may allow multiple "same" presentations of data
        # assert X.shape == data.shape
        return data.reshape(full_sz, initial_data_shape[-1]), (
            initial_shape, full_sz)

    def flatten_labels(labels):
        # Same in flatten_data
        # assert Y.shape == labels.shape
        return labels.flatten()

    line_data, dimen = flatten_data(X)
    line_labels = flatten_labels(Y)

    # Split

    (train, cv, test), (train_labels, cv_labels, test_labels) = \
        split_train_test_data(line_data, line_labels, **config)
    # train
    mean = np.mean(line_data, 0)
    var = np.var(line_data, 0)
    if config.get("normalize"):
        def normalizator(x):
            return (x - mean) / var
    else:
        def normalizator(x):
            return x

    train = normalizator(train)

    classifier = classifier_method()
    clf = classifier.fit(train, train_labels)

    # Cross validate

    if config.get("cv_p", 0):
        # TODO: Implement cross validation and some optimization of parameters
        pass

    # Create predictor

    predictor = Predictor(clf, flatten_data, revert_data_shape, flatten_labels,
                          revert_labels_shape, test, test_labels, normalizator)

    # Test
    # Predict
    predicted_labels = predictor.flatten_labels(predictor.predict(test))

    # evaluate results
    precission = precision_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels)

    test_label_mask_predictions = (
        np.pad(predicted_labels[::-1], (0, full_size - len(predicted_labels)),
               "constant")[::-1]).reshape(initial_label_shape)

    return (precission, recall, f1), predictor, test_label_mask_predictions


def find_closest_date(dates, date):
    return bisect.bisect(dates, date)


def find_index_after_date(dates, date):
    return bisect.bisect(dates, date)


class Reshaper:

    def __init__(self, data_shape):
        self.data_shape = data_shape
        full_sz = 1
        for j in self.data_shape[:-1]:
            full_sz *= j
        self.full_size = full_sz

    def revert_data_shape(self, data):
        return data.reshape(self.data_shape)

    def revert_labels_shape(self, labels):
        return labels.reshape(self.data_shape[:-1])

    def flatten_data(self, data):
        return data.reshape(self.full_size, self.data_shape[:-1])

    def flatten_labels(self, labels):
        return labels.flatten()
