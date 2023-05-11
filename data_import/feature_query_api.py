"""
API for querying feature-values using a database mapping in JSON-format
Schema:
{
    "columns": [
        {
            "name": "Name of the feature",
            "column": "Column containing the feature's values",
            "table": "Table containing the feature's column",
            "idColumn": "Id-Column of the table",
            "type": "Data-type of the features column",
            "classes": ["categorical", "label", "tags", "for", "features", ...]
        },
        ...
    ]
    "timestamp": "String representing creation timestamp of this mapping, used to distinguish mappings"
}
"""
import json
import oracledb

# constants for labels in json-mapping
KEY_FEATURES = "features"
KEY_TIMESTAMP = "timestamp"
KEY_TABLE = "table"
KEY_ID_COLUMN = "idColumn"
KEY_COLUMN = "column"
KEY_DATA_TYPE = "type"
KEY_NAME = "name"
KEY_CLASSES = "classes"


def load(path: str) -> dict:
    """Loads a database mapping from a file in json-format"""
    file = open(path)
    content = file.read()
    file.close()
    return json.loads(content)


def collect_table_data(feature_mapping: dict) -> dict:
    """Creates an alias for every table contained in the mapping returns a dict of type {"table":TableData(...)}"""
    index = 0
    table_data = {}
    for column in feature_mapping[KEY_FEATURES]:
        table = column[KEY_TABLE]
        if table not in table_data:
            # new table -> create new alias and add to result
            table_data[table] = TableData(table, column[KEY_ID_COLUMN], "e" + str(index))
            index = index + 1
    # add the first table's data to key 0
    for table in table_data:
        table_data[0] = table_data[table]
        break

    return table_data


def query_features(connection: oracledb.Connection, feature_mapping: dict) -> list:
    """queries all features in the given mapping"""
    cursor = connection.cursor()
    cursor.prefetchrows = 1000
    cursor.arraysize = 5000

    # create query statement for all features
    query_statement = build_query_statement(feature_mapping)

    # execute statement
    print("querying features...")
    print(query_statement)
    cursor.execute(query_statement)

    # fetch result
    print("fetching...", end="")
    result = []
    while True:
        fetched_rows = cursor.fetchmany()
        if len(fetched_rows) == 0:
            break
        result += fetched_rows
        print("\rfetched:", len(result), "rows", end="")
    print()
    return result


def build_query_statement(feature_mapping: dict) -> str:
    """builds a query statement for all features in the mapping, feature columns can belong to different tables"""
    # get table aliases
    table_data = collect_table_data(feature_mapping)
    # select all features by using their aliases
    query_statement = "SELECT " + ", ".join(
        [table_data[feature[KEY_TABLE]].alias + "." + feature[KEY_COLUMN] for feature in feature_mapping[KEY_FEATURES]]
    )
    # from all tables
    # noinspection PyTypeChecker
    query_statement += " FROM " + ", ".join(
        [table + " " + table_data[table].alias for table in table_data if type(table) is str])
    # if there is more than one table, join tables using their id-column
    if len(table_data) > 2:  # compare to two, since index '0' is always set
        join_table = table_data[0]
        query_statement += " WHERE " + " AND ".join(
            [join_table.alias + "." + join_table.id_column + "=" + table_data[td].alias + "." + table_data[td].id_column
             for td in table_data if td != join_table.table and td != 0]
        )
    return query_statement


def get_timestamp(mapping: dict) -> str:
    """extracts the timestamp-attribute from a mapping"""
    return mapping[KEY_TIMESTAMP]


class TableData:

    def __init__(self, table, id_column, alias):
        self.table = table
        self.id_column = id_column
        self.alias = alias
