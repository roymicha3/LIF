from experiment_manager.environment import Environment
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.pipelines.pipeline import Pipeline

# the pipeline factory logic is already implemented in the parent class
from experiment_manager.pipelines.pipeline_factory import PipelineFactory

# Import all pipelines
from pipeline.training_pipeline import TrainingPipeline


class CustomPipelineFactory(PipelineFactory):
    """
    Factory class for creating pipelines.
    """

    @staticmethod
    def create(name: str, config, env: Environment) -> Pipeline:
        """
        Create an instance of a registered pipeline.
        """
        # the pipeline factory logic is already implemented in the parent class!
        return PipelineFactory.create(name, config, env)