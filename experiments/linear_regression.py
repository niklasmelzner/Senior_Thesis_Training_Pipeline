from task_templates import create_model_template_definition, import_and_prepare_feature_table
from experiment_suite import perform_experiment_tasks
from utils import put_in_file_context

CONNECTION_CONFIG_FILE = "../connection_config.ini"

if __name__ == "__main__":
    labels, features, config = import_and_prepare_feature_table(CONNECTION_CONFIG_FILE)

    experiment_definition = create_model_template_definition(
        sample_size=len(features.feature_values)
    )

    # elastic net tasks for various l1_ratio and alpha values
    tasks = experiment_definition.use_template("linearRegression") \
        .build_tasks()

    # execute tasks
    perform_experiment_tasks(
        tasks=tasks,
        features=features,
        labels=labels,
        suite_name="linear_regression",
        dir_results=put_in_file_context(config["EXPORT"]["result_dir"], CONNECTION_CONFIG_FILE)
    )
# 850 mb per worker
