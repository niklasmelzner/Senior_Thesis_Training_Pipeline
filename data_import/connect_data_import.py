"""
Used to load the feature table, caches loaded data, see :func:`load`.
"""
import oracledb
import data_import.feature_query_api as fqa
from utils import FeatureTable
from utils import timelog
from utils import put_in_file_context
import numpy as np
import configparser
import os

db_connection_parameters = None


def configure_connection(**attributes) -> None:
    """configures the database connection (globally)"""
    global db_connection_parameters
    db_connection_parameters = attributes


def query_cache(feature_mapping: dict, file_name: str) -> None:
    """queries all features into a cache file"""
    # connect to database
    connection = oracledb.connect(**db_connection_parameters)

    # query feature values
    query_result = fqa.query_features(connection, feature_mapping)

    # get feature names
    feature_columns = [column[fqa.KEY_COLUMN] for column in feature_mapping[fqa.KEY_FEATURES]]

    # compress into cache file
    np.savez_compressed(file_name, feature_values=query_result,
                        feature_columns=feature_columns,
                        feature_timestamp=fqa.get_timestamp(feature_mapping))


def load_data(feature_mapping: dict, file_name: str) -> FeatureTable:
    """loads the data for a feature mapping from the cache or if not valid from the database"""
    log = timelog.start()
    file_data = None
    if not os.path.exists(file_name):
        # no cache file -> create one
        print("cache invalid, re-querying data...")
        query_cache(feature_mapping, file_name)
    else:
        # load cache data
        file_data = np.load(file_name, allow_pickle=True)
        if file_data["feature_timestamp"] != feature_mapping["timestamp"]:
            # data out of data -> requery data
            print("cache invalid, re-querying data...")
            query_cache(feature_mapping, file_name)
            file_data = None

    # load cache if it got updates
    if file_data is None:
        file_data = np.load(file_name, allow_pickle=True)

    features_by_columns = {}
    for feature in feature_mapping[fqa.KEY_FEATURES]:
        features_by_columns[feature[fqa.KEY_COLUMN]] = feature

    features = []
    for featureColumn in file_data["feature_columns"]:
        features.append(features_by_columns[featureColumn])

    feature_table = FeatureTable(file_data["feature_values"], feature_configs=features)

    log.log("loaded data in {0}s: " + str(len(feature_table.features)) + " features, " +
            str(len(feature_table.feature_values)) + " data sets")
    return feature_table


def import_config_file(config_file: str) -> tuple:
    """imports a .ini config file containing connection and feature file configuration"""
    config = configparser.ConfigParser()
    config.read(config_file)

    connection_config = config['CONNECTION']

    configure_connection(host=connection_config['host'], port=connection_config['port'], sid=connection_config['sid'],
                         user=connection_config['user'], password=connection_config['password'])

    return put_in_file_context(config['MAPPING']['feature_file'], config_file), \
        put_in_file_context(config['CACHE']['cache_file'], config_file), config


def load_config(feature_file: str = None, config_file: str = None) -> dict:
    if config_file is not None:
        feature_file, cache_file = import_config_file(config_file)

    # load mapping from files
    return fqa.load(feature_file)


def load(feature_file: str = None, cache_file: str = None, config_file: str = None) -> tuple[FeatureTable, any]:
    """loads feature data from the cache/database
    if config_file, feature_file and cache_file are ignored"""
    if config_file is not None:
        feature_file, cache_file, config = import_config_file(config_file)
    else:
        config = None

    # load mapping from files
    features = fqa.load(feature_file)

    # load data
    return load_data(features, cache_file), config
