from task_templates import create_model_template_definition, import_and_prepare_feature_table
from experiment_suite import perform_experiment_tasks
from utils import put_in_file_context

CONNECTION_CONFIG_FILE = "../connection_config.ini"

if __name__ == "__main__":
    labels, features, config = import_and_prepare_feature_table(CONNECTION_CONFIG_FILE)

    experiment_definition = create_model_template_definition(
        sample_size=len(features.feature_values)
    )

    # random forest task for max_tree_depths of 3, 5, 7 and 11 - 3 runs each
    tasks = experiment_definition.use_template("rf") \
        .vary_parameter(max_depth=[3, 5, 7, 11]) \
        .cross_validate("run_1", "run_2", "run_3") \
        .add_parameters(max_features="sqrt", n_estimators=100) \
        .build_tasks()

    # execute tasks
    perform_experiment_tasks(
        tasks=tasks,
        features=features,
        labels=labels,
        suite_name="rf_varying_tree_depth",
        dir_results=put_in_file_context(config["EXPORT"]["result_dir"], CONNECTION_CONFIG_FILE)
    )
# 700-800 mb per worker originally: 500mb
