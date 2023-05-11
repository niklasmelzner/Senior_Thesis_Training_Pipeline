"""
Responsible for executing TrainModelTasks, see perform_experiment_tasks
"""
import os
import pickle
import time

import numpy as np
from sklearn.metrics import _classification
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from utils import FeatureTable
from utils.feature_table import EncodedFeature
from utils.parallelize import compute_in_parallel
from utils.write_csv import write_as_csv, merge_csv_files
from .experiment_definition import TrainModelTask, ExperimentDefinition


def normalize_feature_values(feature_table: FeatureTable, feature_filter: callable) -> StandardScaler:
    """
    Normalizes feature values of the features that match the filter
    """
    binary_feature_filter = [feature_filter(feature) for feature in feature_table.features]

    scaler = StandardScaler()

    if len(feature_table.features) != 0:
        feature_table.feature_values[:, binary_feature_filter] = \
            scaler.fit_transform(feature_table.feature_values[:, binary_feature_filter])

    return scaler


def resample_values(values, target_size):
    """
    Selects a random sample from the given values
    """
    return np.array(values)[np.random.choice(len(values), size=target_size, replace=len(values) < target_size)]


def create_test_data_filter_and_balance_data(features: FeatureTable, labels: FeatureTable, test_ratio: float,
                                             sample_size: int):
    """
    Creates a test data filter with the given ratio and balances data in the training dataset
    (test_data_filter==0) so every class is equally represented
    """
    # create new tables
    features = features.copy()
    labels = labels.copy()

    # create the test and training data filter
    num_test = int(sample_size * test_ratio)
    num_train = sample_size - num_test

    test_data_filter = np.concatenate((np.full(num_test, True),
                                       np.full(len(features.feature_values) - num_test, False)))
    np.random.shuffle(test_data_filter)
    training_data_filter = np.invert(test_data_filter)

    # split features and labels according to the filters
    features_test = FeatureTable(feature_values=features.feature_values[test_data_filter], features=features.features)
    features_train = FeatureTable(feature_values=features.feature_values[training_data_filter],
                                  features=features.features)

    labels_test = FeatureTable(feature_values=labels.feature_values[test_data_filter], features=labels.features)
    labels_train = FeatureTable(feature_values=labels.feature_values[training_data_filter], features=labels.features)

    # count training labels
    classes, counts = np.unique(labels_train.feature_values.flatten(), return_counts=True)

    # calculate target count per class
    target_count_per_class = num_train // len(classes)
    indices_overflow = num_train % len(classes)

    # collect indices for each class
    indices_by_class = {c: [] for c in classes}
    for i in range(len(labels_train.feature_values)):
        indices_by_class[labels_train.feature_values[i, 0]].append(i)

    # resample rows by resampling indices to the target count per class
    i = 0
    for c in indices_by_class:
        indices_by_class[c] = resample_values(indices_by_class[c], target_count_per_class +
                                              (1 if i < indices_overflow else 0))
        i += 1

    # all indexes that shall be contained in training data
    total_indices = np.concatenate([indices_by_class[i] for i in indices_by_class])
    np.random.shuffle(total_indices)

    # select training data rows
    features_train.feature_values = features_train.feature_values[total_indices]
    labels_train.feature_values = labels_train.feature_values[total_indices]

    # reshuffle resampled training data and original test data together
    sample_test_data_filter = np.concatenate((np.full(num_train, False), np.full(num_test, True)))
    np.random.shuffle(sample_test_data_filter)
    sample_training_data_filter = np.invert(sample_test_data_filter)
    features.feature_values = np.empty_like(
        features.feature_values, shape=(len(sample_test_data_filter), features.feature_values.shape[1]))
    labels.feature_values = np.empty_like(
        labels.feature_values, shape=(len(sample_test_data_filter), labels.feature_values.shape[1]))

    features.feature_values[sample_test_data_filter] = features_test.feature_values
    features.feature_values[sample_training_data_filter] = features_train.feature_values

    labels.feature_values[sample_test_data_filter] = labels_test.feature_values
    labels.feature_values[sample_training_data_filter] = labels_train.feature_values

    return features, labels, sample_test_data_filter


def one_hot_encode_features(table: FeatureTable, feature_filter: callable):
    """
    Encodes features based on the given filter
    """
    features_to_encode = table[feature_filter]
    other_features = table[lambda feature: not feature_filter(feature)]

    if len(features_to_encode.features) == 0:
        return other_features

    encoder = OneHotEncoder()
    # transform feature values
    encoded_feature_data = encoder.fit_transform(features_to_encode.feature_values).toarray()

    # get new feature names
    encoded_feature_names = encoder.get_feature_names_out(
        [feature.column for feature in features_to_encode.features])

    # create EncodedFeatures objects for all new features referencing the original Feature object
    encoded_feature_values = np.concatenate(encoder.categories_)
    encoded_features = []
    for name_index in range(len(encoded_feature_names)):
        name = encoded_feature_names[name_index]
        encoded_feature = None
        for feature in features_to_encode.features:
            if str(name).startswith(feature.column):
                encoded_feature = EncodedFeature(feature, encoded_feature_values[name_index])
                encoded_feature.name = name
                encoded_feature.column = name
                break

        encoded_features.append(encoded_feature)

    # create a table for the encoded data
    encoded_feature_table = FeatureTable(feature_values=encoded_feature_data,
                                         features=np.array(encoded_features))
    # merge with non encoded features
    return other_features + encoded_feature_table


class ModelTrainingResult:
    """
    Summarizes the result of training a model
    """
    RESULT_KEY_SCORE_ACCURACY = "accuracy"
    RESULT_KEY_SCORE_PRECISION = "precision"
    RESULT_KEY_SCORE_RECALL = "recall"
    RESULT_KEY_SCORE_F1 = "f1"
    RESULT_KEY_T_TRAIN = "t_train"
    RESULT_KEY_T_PRED = "t_predict"
    RESULT_KEY_SCORE_NUM_FEATURES = "features"

    def __init__(self, model, y_test, y_pred, features: list[str], labels: list[str]):
        self.model = model
        self.y_test = y_test
        self.y_pred = y_pred
        self.features = features
        self.labels = labels
        self.scores = {ModelTrainingResult.RESULT_KEY_SCORE_NUM_FEATURES: len(features)}

    def add_scores(self, accuracy, recall, precision, f1, dt_train, dt_pred):
        mapping = {
            ModelTrainingResult.RESULT_KEY_SCORE_ACCURACY: accuracy,
            ModelTrainingResult.RESULT_KEY_SCORE_RECALL: recall,
            ModelTrainingResult.RESULT_KEY_SCORE_PRECISION: precision,
            ModelTrainingResult.RESULT_KEY_SCORE_F1: f1,
            ModelTrainingResult.RESULT_KEY_T_TRAIN: dt_train,
            ModelTrainingResult.RESULT_KEY_T_PRED: dt_pred
        }
        for key in mapping:
            if mapping[key] is None:
                continue
            self.scores[key] = mapping[key]
        return self


def train_model(model_type, features: FeatureTable, labels: FeatureTable, sample_size: int, model_params: dict,
                test_ratio: float, normalize_features: callable = None,
                features_to_one_hot_encode: callable = None) -> ModelTrainingResult:
    """
    Trains the given model, creates the test data filter, balances data, normalizes feature values,
    encodes features, calculates scores and returns a ModelTrainingResult
    """
    if sample_size is None:
        sample_size = len(features.feature_values)

    # balance features, create test data filter
    features, labels, test_data_filter = create_test_data_filter_and_balance_data(features, labels,
                                                                                  test_ratio, sample_size)

    # normalize feature values (if requested)
    if normalize_features is not None and len(features[normalize_features].features) != 0:
        normalize_feature_values(features, normalize_features)

    # one hot encode features (if requested)
    if features_to_one_hot_encode is not None:
        features = one_hot_encode_features(features, features_to_one_hot_encode)

    # instantiate model
    model = model_type(**model_params)

    # split data using test_data_filter
    test_data_filter_inverse = [not i for i in test_data_filter]
    x_train = features.feature_values[test_data_filter_inverse]
    x_test = features.feature_values[test_data_filter]
    y_train = labels.feature_values[test_data_filter_inverse]
    y_test = labels.feature_values[test_data_filter]

    # train the model
    t_train_start = time.time()
    model.fit(x_train, y_train)
    dt_train = time.time() - t_train_start

    # predict a results
    t_pred_start = time.time()
    y_pred = model.predict(x_test)
    dt_pred = time.time() - t_pred_start

    # calculate scores
    accuracy_score = _classification.accuracy_score(y_test, y_pred)
    recall_score = _classification.recall_score(y_test, y_pred, average="macro")
    precision_score = _classification.precision_score(y_test, y_pred, average="macro")
    f1_score = _classification.f1_score(y_test, y_pred, average="macro")

    return ModelTrainingResult(model, y_test, y_pred,
                               [feature.column for feature in features.features],
                               [label.column for label in labels.features]
                               ) \
        .add_scores(accuracy=accuracy_score, recall=recall_score, precision=precision_score,
                    f1=f1_score, dt_train=dt_train, dt_pred=dt_pred)


def run_experiment(task: TrainModelTask, features: FeatureTable,
                   labels: FeatureTable):
    """
    Trans a model using train_model for each label, handles feature filtering using task.feature_filter
    """
    results = {}
    for label in labels.features:
        print("tasks:", task.name, "label:", label.column)

        if label.column in task.feature_filter:
            filtered_features = features[lambda feature: feature.column in task.feature_filter[label.column]]
            print("filtered features,", len(filtered_features.features), "remaining")
        else:
            filtered_features = features

        results[label] = train_model(
            model_type=task.model, features=filtered_features, labels=labels[label],
            model_params=task.params,
            sample_size=task.experiment_definition.sample_size,
            test_ratio=task.experiment_definition.test_ratio,
            normalize_features=task.experiment_definition.normalize_features,
            features_to_one_hot_encode=task.experiment_definition.one_hot_encode_features
        )
    return results


def perform_experiment_task_internal(
        index, tasks: list[TrainModelTask],
        features: FeatureTable,
        labels: FeatureTable,
        dir_results: str) -> dict:
    """
    Executes a task using run_experiment and serializes the result
    """
    task = tasks[index]

    training_results = run_experiment(task, features, labels)

    # serialize the object with all available result data
    path_raw_results = task.name + ".pkl"
    with open(dir_results + "/" + path_raw_results, 'wb') as f:
        pickle.dump(training_results, f)

    # write training scores and model parameters to csv
    path_scores = task.name + "_scores.csv"
    score_rows = []
    for label in training_results:
        label_result = training_results[label]
        score_rows.append(
            {"task_name": task.name,
             "label": label.column} |
            label_result.scores
        )
    write_as_csv(score_rows, dir_results + "/" + path_scores)

    files_to_summarize = {"scores": dir_results + "/" + path_scores}

    # process result extraction tasks
    result_extraction_tasks = task.get_additional_tasks(ExperimentDefinition.TASK_EXTRACT_RESULTS_CSV)
    for extraction_task in result_extraction_tasks:
        rows = []
        # create table using extraction task
        for label in training_results:
            additional_data = {"task_name": task.name, "label": label.column}

            task_rows = extraction_task["task"](training_results[label])
            rows += [row | additional_data for row in task_rows]
        # serialize result
        target_file = dir_results + "/" + task.name + "_" + extraction_task["tag"] + ".csv"
        write_as_csv(rows, target_file)
        # add to summarization tasks if requested
        if "summarize" in extraction_task and extraction_task["summarize"]:
            files_to_summarize[extraction_task["tag"]] = target_file
    return files_to_summarize


def perform_experiment_tasks(
        tasks: list[TrainModelTask],
        features: FeatureTable,
        labels: FeatureTable,
        suite_name: str,
        dir_results: str,
        n_workers=-1):
    """
    Executes the given tasks in parallel using perform_experiment_task_internal
    Serializes the result to dir_results
    """
    dir_results += "/" + suite_name + "_" + str(time.time())

    if not os.path.exists(dir_results):
        os.mkdir(dir_results)

    t_start = time.time()

    # create a .csv file containing parameters of all tasks
    document_tasks(dir_results, tasks)

    # execute tasks
    files_to_summarize = compute_in_parallel([i for i in range(len(tasks))], perform_experiment_task_internal,
                                             args={"tasks": tasks, "dir_results": dir_results,
                                                   "features": features, "labels": labels}, n_workers=n_workers)

    # summarize files among tasks
    summarize_files(dir_results, files_to_summarize)

    print("trained all models in", (time.time() - t_start), "seconds")
    print("wrote results to", os.path.abspath(dir_results))


def document_tasks(dir_results: str, tasks: list[TrainModelTask]):
    """
    Serializes training information for each task
    """
    rows = []
    for task in tasks:
        rows.append({"task_name": task.name, "model": task.model.__name__} | task.params)
    write_as_csv(rows, dir_results + "/task_definition.csv")


def summarize_files(dir_results: str, files_to_summarize: dict):
    """
    Merges multiple .csv files to summarize results
    """
    files_by_tag = {}
    for files in files_to_summarize:
        for tag in files:
            if tag in files_by_tag:
                files_by_tag[tag].append(files[tag])
            else:
                files_by_tag[tag] = [files[tag]]
    for tag in files_by_tag:
        merge_csv_files(files_by_tag[tag], dir_results + "/all_" + tag + ".csv")
