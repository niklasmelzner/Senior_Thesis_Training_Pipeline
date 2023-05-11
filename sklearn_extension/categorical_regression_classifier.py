"""
Combines scikit-learn regression models for each feature class
by choosing the model with the highest prediction confidence for each class
"""
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class CategoricalRegressionClassifier:

    def __init__(self, model_type, **model_params):
        self.model_type = model_type
        self.model_params = model_params
        # to encode labels
        self.label_encoder = OneHotEncoder()
        self.models = []
        self.trained = False
        self.result_dtype = None

    @staticmethod
    def calculate_normalization_filter(column: np.ndarray):
        """
        creates filter that, when applied to data, produces an even distribution of positives and negatives
        """
        negatives = column[column == 0]
        positives = column[column == 1]

        min_count = min(len(negatives), len(positives))
        min_count_neg = min(len(negatives), min_count)
        negatives_normalization_filter = np.concatenate(
            (np.ones(min_count_neg), np.zeros(len(negatives) - min_count_neg)))
        positives_normalization_filter = np.concatenate(
            (np.ones(min_count), np.zeros(len(positives) - min_count)))
        np.random.shuffle(negatives_normalization_filter)
        np.random.shuffle(positives_normalization_filter)

        normalization_filter = np.full(len(column), False)
        index_positives = 0
        index_negatives = 0
        for i in range(0, len(column)):
            if column[i] == 1:
                normalization_filter[i] = positives_normalization_filter[index_positives] == 1
                index_positives += 1
            else:
                normalization_filter[i] = negatives_normalization_filter[index_negatives] == 1
                index_negatives += 1
        return normalization_filter

    def score(self, x, y):
        y_predict = self.predict(x)
        return (np.count_nonzero(y.flatten() == y_predict)) / np.count_nonzero(y_predict != None)

    def fit(self, x: np.ndarray, y: np.ndarray):
        if self.trained:
            raise Exception("model can only be trained once :(")
        self.trained = True
        result = self.label_encoder.fit_transform(y).toarray()
        self.result_dtype = y.dtype
        # train models for each encoded column
        for iColumn in range(result.shape[1]):
            print("fit " + str(iColumn + 1) + "/" + str(result.shape[1]))
            column = result[:, iColumn].flatten()

            # filter data for normalization
            normalization_filter = CategoricalRegressionClassifier.calculate_normalization_filter(column)

            x_test = x[normalization_filter]
            y_test = column[normalization_filter]

            # train model
            model = self.model_type(**self.model_params)
            model.fit(x_test, y_test)

            self.models.append((model, self.label_encoder.categories_[0][iColumn]))

    def predict(self, x):
        if not self.trained:
            raise Exception("not trained yet :(")

        # selects label class with the smallest distance
        min_distances = np.empty(len(x))
        result = np.empty(len(x), self.result_dtype)
        first = True
        for model, category in self.models:
            y_pred = model.predict(x)

            distances = np.abs(1 - y_pred)

            if first:
                min_distances = distances
                result[:] = category
                first = False
            else:
                replace_filter = distances < min_distances
                min_distances[replace_filter] = distances[replace_filter]
                result[replace_filter] = category

        return result
