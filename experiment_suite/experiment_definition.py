"""
Factory to construct experiments.
Each experiment definition has a unique name that identifies it, references a model and stores parameters for this model.
Example usage:

experiment_definition = ExperimentDefinition()\
    .define_model_template("template", RandomForestClassifier,
        n_estimators=5, class_weight="balanced"
    )

tasks = experiment_definition \
    .use_template("template") \
    .vary_parameter(max_depth=[3, 5, 7, 11]) \
    .cross_validate("run_1", "run_2", "run_3") \
    .add_parameters(max_features="sqrt") \
    .build_tasks()

"""
from typing import TypeVar, Union

SelfExperimentDefinition = TypeVar("SelfExperimentDefinition", bound="ExperimentDefinition")
SelfTaskFactory = TypeVar("SelfTaskFactory", bound="TaskFactory")


class TrainModelTemplate:
    """
    Named template that stores a model and corresponding parameters
    """

    def __init__(self, task_name: str, model, model_params):
        self.name = task_name
        self.model = model
        self.model_params = model_params
        self.additional_tasks = []

    def register_task(self, task_type: str, **parameters):
        self.additional_tasks.append({"task_type": task_type} | parameters)


class TrainModelTask:
    """
    Named task that represents training of a single model instance
    """

    def __init__(self, experiment_definition: SelfExperimentDefinition, task_name: Union[str, None], model,
                 params: dict, additional_tasks: list, feature_filter: dict):
        self.experiment_definition = experiment_definition
        self.name = task_name
        self.model = model
        self.params = params
        self.additional_tasks = additional_tasks
        self.feature_filter = feature_filter

    def add_feature_filter(self, feature_filter: dict):
        self.feature_filter |= feature_filter

    def add_parameters(self, params):
        self.params |= params

    def get_additional_tasks(self, task_type: str):
        return [task for task in self.additional_tasks if task["task_type"] == task_type]

    def copy(self):
        return TrainModelTask(self.experiment_definition, self.name, self.model, self.params.copy(),
                              self.additional_tasks, self.feature_filter)

    def append_to_name(self, name):
        self.name = self.name + "_" + name if self.name is not None else name

    def __repr__(self):
        return "TrainModelTask[" + str(self.name) + ": " + str(self.model) + "]" + str(self.params)


class TaskFactory:
    """
    Factory for TrainModelTask from an ExperimentDefinition,
    provides various functions to add parameters, run tasks multiple times or apply templates
    """

    def __init__(self, parent: SelfTaskFactory):
        self.parent = parent

    def use_template(self, template_name: str) -> SelfTaskFactory:
        """
        Copies the data from a template in the original ExperimentDefinition to all tasks in the pipeline
        """
        return TemplateTaskFactory(self, template_name)

    def vary_parameter(self, **parameter) -> SelfTaskFactory:
        """
        Creates a new task in the pipeline for every parameter provided
        """
        if len(parameter) > 1:
            raise Exception("Only one parameter allowed!")
        name, values = [(key, parameter[key]) for key in parameter][0]
        return VaryParameterTaskFactory(self, name, values)

    def add_parameters(self, **parameters) -> SelfTaskFactory:
        """
        Adds parameters to each task in the pipeline
        """
        return SimpleTransformationTaskFactory(
            self,
            lambda task: task.add_parameters(parameters)
        )

    def cross_validate(self, *names) -> SelfTaskFactory:
        """
        Creates len(names) tasks for each task in the pipeline (runs a task multiple times)
        """
        return CrossValidateTaskFactory(self, names)

    def build_tasks(self) -> list[TrainModelTask]:
        """
        End of the pipeline, builds all tasks
        """
        return self.create_tasks()

    def create_tasks(self) -> list[TrainModelTask]:
        """
        Provides all paths at this step in the pipeline
        """
        raise NotImplementedError()

    def tags(self, *tags):
        """
        Adds tags to the task's name
        """
        return SimpleTransformationTaskFactory(
            self,
            lambda task: task.append_to_name("_".join(tags))
        )

    def filter_features(self, filter: dict):
        """
        Adds a feature filter to the task
        """
        return SimpleTransformationTaskFactory(
            self,
            lambda task: task.add_feature_filter(filter)
        )

    def get_experiment_definition(self) -> SelfExperimentDefinition:
        """
        Returns the experiment definition, this pipeline was started on
        """
        current = self
        while type(current) != ExperimentDefinition:
            current = current.parent
        return current


class SimpleTransformationTaskFactory(TaskFactory):
    """
    Implements the pipeline step for a simple transformation expressed by a callable
    """

    def __init__(self, parent: SelfTaskFactory, transformation: callable):
        super().__init__(parent)
        self.transformation = transformation

    def create_tasks(self) -> list[TrainModelTask]:
        parent_tasks = self.parent.create_tasks()
        tasks = []
        for parent_task in parent_tasks:
            task = parent_task.copy()
            self.transformation(task)
            tasks.append(task)
        return tasks


class CrossValidateTaskFactory(TaskFactory):
    """
    Implements the pipeline step for TaskFactory.cross_validate
    """

    def __init__(self, parent: SelfTaskFactory, names: tuple):
        super().__init__(parent)
        self.task_names = names

    def create_tasks(self):
        parent_tasks = self.parent.create_tasks()
        tasks = []
        for parent_task in parent_tasks:
            for name in self.task_names:
                task = parent_task.copy()
                task.append_to_name(name)
                tasks.append(task)
        return tasks


class VaryParameterTaskFactory(TaskFactory):
    """
    Implements the pipeline step for TaskFactory.vary_parameter
    """

    def __init__(self, parent: SelfTaskFactory, parameter_name: str, values: list):
        super().__init__(parent)
        self.parameter_name = parameter_name
        self.parameter_values = values

    def create_tasks(self):
        parent_tasks = self.parent.create_tasks()
        tasks = []
        for parent_task in parent_tasks:
            for value in self.parameter_values:
                task = parent_task.copy()
                task.params[self.parameter_name] = value
                task.append_to_name(self.parameter_name)
                task.append_to_name(str(value))
                tasks.append(task)
        return tasks


class TemplateTaskFactory(TaskFactory):
    """
    Implements the pipeline step for TaskFactory.use_template
    """

    def __init__(self, parent: SelfTaskFactory, template_name: str):
        super().__init__(parent)
        self.template_name = template_name

    def create_tasks(self):
        template = self.get_experiment_definition().templates[self.template_name]
        template: TrainModelTemplate
        parent_tasks = self.parent.create_tasks()
        tasks = []
        for parent_task in parent_tasks:
            task = parent_task.copy()
            task.model = template.model
            task.append_to_name(template.name)
            task.params |= template.model_params
            task.additional_tasks += template.additional_tasks
            tasks.append(task)
        return tasks


class ExperimentDefinition(TaskFactory):
    """
    Defines multiple experiments
    """
    TASK_EXTRACT_RESULTS_CSV = "TASK_EXTRACT_RESULTS"

    def __init__(self, sample_size: int = None, test_ratio: float = 0.25,
                 normalize_features: callable = None,
                 one_hot_encode_features: callable = None):
        super().__init__(self)
        self.normalize_features = normalize_features
        self.one_hot_encode_features = one_hot_encode_features
        self.test_ratio = test_ratio
        self.sample_size = sample_size
        self.templates = {}

    def define_model_template(self, name, model, **model_params) -> TrainModelTemplate:
        """
        Adds a template that can be applied in :func:'~TaskFactory.use_templates'
        """
        model_template = TrainModelTemplate(name, model, model_params)
        self.templates[name] = model_template
        return model_template

    def create_tasks(self):
        """
        One empty task as the start of the pipeline
        """
        return [TrainModelTask(self, None, None, {}, [], {})]
