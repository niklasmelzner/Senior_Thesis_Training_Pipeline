"""
Uses the feature_query_api to query data from a connect database
Manages the feature query cache
"""
import data_import.feature_query_api as fqa
import numpy as np
from typing import Union, TypeVar

SelfFeatureTable = TypeVar("SelfFeatureTable", bound="FeatureTable")


class Feature:
    """
    Container for feature attributes
    """
    is_encoded = False

    def __init__(self, config=None, feature=None):
        if config is not None:
            self.name = config[fqa.KEY_NAME]
            self.classes = config[fqa.KEY_CLASSES]
            self.column = config[fqa.KEY_COLUMN]
        elif feature is not None:
            self.name = feature.name
            self.classes = feature.classes
            self.column = feature.column

    def __repr__(self):
        if len(self.classes) > 0:
            return self.name + str(self.classes)
        else:
            return self.name


class EncodedFeature(Feature):
    """
    For encoded features, references the original feature and the feature class this
    encoded feature represents
    """
    is_encoded = True

    def __init__(self, parent: Feature, category: object):
        super().__init__(feature=parent)
        self.category = category
        self.parent = parent

    def __repr__(self):
        return super().__repr__() + "@" + str(self.category)


class FeatureTable:
    """
    Combines features and feature values based on numpy arras.
    Should be replaced by panda arrays in the future!
    """

    def __init__(self, feature_values=Union[list[list], np.ndarray], feature_configs: list[dict[str:any]] = None,
                 features: np.ndarray[Feature] = None):
        self.feature_values = feature_values
        if features is None:
            self.features = []
            for config in feature_configs:
                self.features.append(Feature(config=config))
            self.features = np.array(self.features)
        else:
            self.features = features

        self.index_by_feature = {}
        for i in range(len(self.features)):
            self.index_by_feature[self.features[i]] = i

    def copy(self):
        return FeatureTable(feature_values=np.copy(self.feature_values), features=np.copy(self.features))

    def __getitem__(self, index: Union[int, Feature, EncodedFeature, slice, callable]) -> SelfFeatureTable:
        """
        Allows filtering by index, range, feature or function
        """
        if type(index) == int:
            index = self.features[index]
        if type(index).__name__ == 'function':
            return self.filter_features(index)
        elif type(index) == Feature or type(index) == EncodedFeature:
            filter = [False for _ in range(len(self.features))]
            filter[self.index_by_feature[index]] = True
            return FeatureTable(feature_values=self.feature_values[:, filter],
                                features=np.array([index]))
        elif type(index) == slice:
            return FeatureTable(feature_values=self.feature_values[:, index],
                                features=self.features[index])
        else:
            raise TypeError(index, type(index))

    def __add__(self, other: SelfFeatureTable):
        """
        Concatenates the columns of two feature tables, assumes same number of rows and no overlapping columns
        """
        assert type(other) is FeatureTable
        # noinspection PyTypeChecker

        return FeatureTable(
            feature_values=np.concatenate([self.feature_values, other.feature_values], axis=1),
            features=np.concatenate([self.features, other.features])
        )

    def transform_dataset(self, transformation: callable):
        """
        Applies a transformation on the feature values
        """
        return FeatureTable(feature_values=transformation(self.feature_values), features=self.features)

    def filter_features(self, filter: callable):
        """
        Filters features based on a criterion
        """
        filter_result = []
        for feature in self.features:
            filter_result.append(filter(feature))
        return FeatureTable(feature_values=self.feature_values[:, filter_result], features=self.features[filter_result])

    def filter_rows(self, filter):
        return FeatureTable(feature_values=self.feature_values[filter], features=self.features)

    def export_as_csv(self, path: str):
        """
        Exports the table ot a .csv file
        """
        with open(path, "w") as f:
            f.write(",".join([feature.column for feature in self.features]))
            # noinspection PyTypeChecker
            for row in self.feature_values:
                f.write("\n")
                f.write(",".join(['"' + str(v) + '"' for v in row]))
