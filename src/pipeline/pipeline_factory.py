from experiment_manager.environment import Environment
from experiment_manager.common.factory import Factory
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.pipelines.pipeline import Pipeline

# Import all pipelines
from pipeline.training_pipeline import TrainingPipeline


class PipelineFactory(Factory):
    """
    Factory class for creating pipelines.
    """

    @staticmethod
    def create(name: str, config, env: Environment) -> Pipeline:
        """
        Create an instance of a registered pipeline.
        """
        return YAMLSerializable.get_by_name(name).from_config(config, env)