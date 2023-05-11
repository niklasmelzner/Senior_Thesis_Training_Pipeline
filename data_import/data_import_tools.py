from data_import import connect_data_import
from utils import FeatureTable
import pandas as pd

def import_data(import_config_file:str) -> tuple[FeatureTable, any]:

    """Handles the data import process, removes rows with NA values"""
    feature_table, import_config = connect_data_import.load(config_file=import_config_file)

    # remove empty rows
    size_before = len(feature_table.feature_values)
    feature_table = feature_table.transform_dataset(lambda values: values[~pd.isna(values).any(axis=1)])

    print("removed " + str(size_before - len(feature_table.feature_values)) + " instances that contained empty values",
          "(" + str(len(feature_table.feature_values)) + " instances remain)")

    return feature_table, import_config


def encode_int_as_string(feature_table: FeatureTable, criterion: callable) -> FeatureTable:
    """Encodes columns selected by the criterion first as int, then as string"""
    features_to_encode, other_features = split_table(feature_table, criterion)
    features_to_encode.feature_values = features_to_encode.feature_values.astype("int").astype("str")
    return features_to_encode + other_features


def split_table(feature_table: FeatureTable, criterion: callable) -> tuple[FeatureTable, FeatureTable]:
    """Splits a table by a criterion"""
    return feature_table[criterion], feature_table[lambda feature: not criterion(feature)]
