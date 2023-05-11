"""
Defines template experiments and adds tasks for extracting further information
(feature importance and feature coefficients) from training results
"""
from sklearn.ensemble import RandomForestClassifier

from data_import import data_import_tools
from experiment_suite import ExperimentDefinition
from experiment_suite.perform_experiment import ModelTrainingResult
from sklearn_extension import CategoricalRegressionClassifier
from sklearn.linear_model import ElasticNet, LinearRegression

from utils import FeatureTable


def import_and_prepare_feature_table(config_file: str) -> tuple[FeatureTable, FeatureTable, any]:
    """
    Imports a feature table using the config file, encodes the label "#Cylinders" and splits the table into
    labels and features
    :return labels, features, config loaded from config_file
    """
    # all features
    feature_table, config = data_import_tools.import_data(config_file)

    # encode label "AnzahlZylinder"
    feature_table = data_import_tools.encode_int_as_string(
        feature_table, lambda feature: "Zylinder" in feature.column
    )

    # split into features and labels based on tags
    labels, features = data_import_tools.split_table(
        feature_table, lambda feature: "label" in feature.classes
    )
    return labels, features, config


def numerical_feature_filter(feature):
    return not categorical_feature_filter(feature)


def categorical_feature_filter(feature):
    return "categorical" in feature.classes


def create_model_template_definition(sample_size: int) -> ExperimentDefinition:
    """
    Creates templates for all tasks
    """
    experiment_definition = ExperimentDefinition(
        sample_size=sample_size,
        test_ratio=0.25,
        # normalize numerical features
        normalize_features=numerical_feature_filter,
        # encode categorical features
        one_hot_encode_features=categorical_feature_filter
    )
    add_rf_template(experiment_definition)
    add_el_template(experiment_definition)
    add_lr_template(experiment_definition)
    return experiment_definition


def add_rf_template(experiment_definition: ExperimentDefinition):
    """
    Sets default parameters for Random Forest models and adds a result interpretation task
    extracting feature importance from the model
    """
    experiment_definition.define_model_template(
        "rf", RandomForestClassifier,
        n_estimators=100, min_samples_split=2, min_samples_leaf=2,
        max_features="sqrt", max_depth=5, class_weight="balanced",
        bootstrap=True, verbose=0, n_jobs=1
    ).register_task(ExperimentDefinition.TASK_EXTRACT_RESULTS_CSV, tag="feature_importance",
                    task=extract_rf_feature_importance, summarize=True)


def extract_rf_feature_importance(training_result: ModelTrainingResult):
    """
    Creates a table containing the importance of each feature in the result
    """
    model = training_result.model
    model: RandomForestClassifier
    rows = []
    importances = model.feature_importances_
    for i in range(len(training_result.features)):
        rows.append({"feature": training_result.features[i], "importance": importances[i]})

    return rows


def add_el_template(experiment_definition: ExperimentDefinition):
    """
    Sets default parameters for random Elastic Net and adds a result interpretation task
    extracting feature coefficients from the model
    """
    experiment_definition.define_model_template(
        "elasticNet", CategoricalRegressionClassifier, model_type=ElasticNet,
        alpha=0.003, l1_ratio=0.5, max_iter=5000,
        selection="random", warm_start=True
    ).register_task(ExperimentDefinition.TASK_EXTRACT_RESULTS_CSV, tag="coeffs",
                    task=extract_el_coefficients, summarize=True)


def extract_el_coefficients(training_result: ModelTrainingResult):
    """
    Creates a table containing the coefficients of each feature for each label value in the result
    """
    model = training_result.model
    model: CategoricalRegressionClassifier
    features = training_result.features
    rows = []
    for el_model, category in model.models:
        el_model: ElasticNet
        coeffs = el_model.coef_
        for i in range(len(features)):
            if coeffs[i] == 0:
                continue
            rows.append({"value": category, "feature": features[i], "coeff": coeffs[i]})
    return rows


def add_lr_template(experiment_definition: ExperimentDefinition):
    """
    Sets default parameters for Linear Regression tasks
    """
    experiment_definition.define_model_template(
        "linearRegression", CategoricalRegressionClassifier, model_type=LinearRegression
    ).register_task(ExperimentDefinition.TASK_EXTRACT_RESULTS_CSV, tag="coeffs",
                    task=extract_el_coefficients, summarize=True)
