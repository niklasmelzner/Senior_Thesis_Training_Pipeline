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
    tasks = experiment_definition.use_template("elasticNet") \
        .add_parameters(max_iter=3000) \
        .vary_parameter(l1_ratio=[0, 0.2, 0.4, 0.6, 0.8, 1.0]) \
        .vary_parameter(alpha=[0.001, 0.003, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2]) \
        .build_tasks()

    # execute tasks
    perform_experiment_tasks(
        tasks=tasks,
        features=features,
        labels=labels,
        suite_name="elastic_net",
        dir_results=put_in_file_context(config["EXPORT"]["result_dir"], CONNECTION_CONFIG_FILE)
    )
# 850 mb per worker
