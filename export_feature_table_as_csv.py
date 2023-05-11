"""
exports feature values and mapping as .csv
"""
from data_import import data_import_tools
from utils import write_as_csv
from utils import put_in_file_context

CONNECTION_CONFIG_FILE = "connection_config.ini"

if __name__ == "__main__":
    feature_table, import_config = data_import_tools.import_data(CONNECTION_CONFIG_FILE)

    # export table data
    feature_table.export_as_csv(
        put_in_file_context(import_config["EXPORT"]["feature_table_file"], CONNECTION_CONFIG_FILE))

    # export mapping
    mapping_rows = [
        {"column": feature.column, "name": feature.name, "classes": " ".join(feature.classes)}
        for feature in feature_table.features
    ]

    write_as_csv(mapping_rows,
                 put_in_file_context(import_config["EXPORT"]["feature_mapping_file"], CONNECTION_CONFIG_FILE))
