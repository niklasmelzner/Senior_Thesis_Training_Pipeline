from experiment_suite import perform_experiment_tasks
from task_templates import create_model_template_definition, import_and_prepare_feature_table
import pandas as pd

from utils import put_in_file_context

CONNECTION_CONFIG_FILE = "../connection_config.ini"
EL_COEFFS_SOURCE = "../results/elastic_net_1683685255.3009014/all_coeffs.csv"


def create_el_coeff_based_feature_filter(data: pd.DataFrame) -> dict:
    """
    Creates a feature filter of form {"label":["feature_1", "feature_2", ...]}
    for all labels contained in the dataframe using the features used by Elastic Net
    """
    filter = {}
    for label in pd.unique(data["label"]):
        label_data = data[(data["label"] == label) & (data["coeff"] > 0.000001)]
        filter[label] = pd.unique(label_data["feature"])

    return filter


if __name__ == "__main__":
    labels, features, config = import_and_prepare_feature_table(CONNECTION_CONFIG_FILE)

    experiment_definition = create_model_template_definition(
        sample_size=len(features.feature_values)
    )

    el_data = pd.read_csv(EL_COEFFS_SOURCE, encoding="windows-1252")
    # extract tags "max_depth" and "run" from task_name
    el_data[["l1_ratio", "alpha"]] = el_data["task_name"] \
        .str.extract("elasticNet_l1_ratio_([0-9.]+)_alpha_([0-9.]+)")
    el_data["l1_ratio"] = el_data["l1_ratio"].apply(float)
    el_data["alpha"] = el_data["alpha"].apply(float)

    # template for tasks
    task_builder = experiment_definition.use_template("rf") \
        .add_parameters(max_features="sqrt", n_estimators=100, max_depth=5)

    tasks = []
    for l1_ratio in pd.unique(el_data["l1_ratio"]):
        for alpha in pd.unique(el_data["alpha"]):
            task_data = el_data[(el_data["l1_ratio"] == l1_ratio) & (el_data["alpha"] == alpha)]

            # create a filter based in the feature_ratio
            filter = create_el_coeff_based_feature_filter(task_data)

            # add tags and feature filter to template
            tasks += task_builder.tags("l1_ratio_" + str(l1_ratio), "alpha_" + str(alpha)) \
                .filter_features(filter) \
                .build_tasks()

    # execute tasks
    perform_experiment_tasks(
        tasks=tasks,
        features=features,
        labels=labels,
        suite_name="rf_el_feature_set",
        dir_results=put_in_file_context(config["EXPORT"]["result_dir"], CONNECTION_CONFIG_FILE)
    )
# 500 to 750 mb, originally 500
