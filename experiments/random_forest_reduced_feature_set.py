from experiment_suite import perform_experiment_tasks
from task_templates import create_model_template_definition, import_and_prepare_feature_table
import pandas as pd

from utils import put_in_file_context

CONNECTION_CONFIG_FILE = "../connection_config.ini"
RF_FEATURE_IMPORTANCE_SOURCE = "../results/rf_varying_tree_depth_1683673279.730491/all_feature_importance.csv"


def create_importance_based_feature_filter(data: pd.DataFrame, n_features: int):
    """
    Creates a feature filter of form {"label":["feature_1", "feature_2", ...]}
    for all labels contained in the dataframe using the top n features sorted by importance
    """
    filter = {}
    for label in pd.unique(data["label"]):
        sorted_features = data[data["label"] == label].sort_values(by="importance", ascending=False)["feature"]
        sorted_features = sorted_features[:n_features]
        filter[label] = sorted_features.tolist()
    return filter


if __name__ == "__main__":
    labels, features, config = import_and_prepare_feature_table(CONNECTION_CONFIG_FILE)

    experiment_definition = create_model_template_definition(
        sample_size=len(features.feature_values)
    )

    rf_data = pd.read_csv(RF_FEATURE_IMPORTANCE_SOURCE)
    # extract tags "max_depth" and "run" from task_name
    rf_data[["max_depth", "run"]] = rf_data["task_name"].str.extract("rf_max_depth_([0-9]+)_run_([0-9]+)")
    rf_data["max_depth"] = rf_data["max_depth"].apply(int)
    rf_data["run"] = rf_data["run"].apply(int)

    # only use data with max_depth=5
    rf_data = rf_data[rf_data["max_depth"] == 5]

    # template for tasks
    task_builder = experiment_definition.use_template("rf") \
        .add_parameters(max_features="sqrt", n_estimators=100, max_depth=5)

    tasks = []
    for task_run in pd.unique(rf_data["run"]):
        task_data = rf_data[rf_data["run"] == task_run]

        for ratio in [0.75, 0.5, 0.25, 0.1]:
            # create a filter based in the feature_ratio
            filter = create_importance_based_feature_filter(task_data, n_features=int(ratio * len(features.features)))

            # add tags and feature filter to template
            tasks += task_builder.tags("target_run_" + str(task_run), "ratio_" + str(ratio)) \
                .filter_features(filter) \
                .build_tasks()

    # execute tasks
    perform_experiment_tasks(
        tasks=tasks,
        features=features,
        labels=labels,
        suite_name="rf_reduced_feature_set",
        dir_results=put_in_file_context(config["EXPORT"]["result_dir"], CONNECTION_CONFIG_FILE)
    )
# 500 to 750 mb, originally 500
