from .feature_table import FeatureTable
import utils.timelog as timelog
from .write_csv import *
import os

def put_in_file_context(file: str, context_file: str):
    if os.path.isabs(file):
        return file
    return os.path.abspath(os.path.dirname(context_file)) + "/" + file
