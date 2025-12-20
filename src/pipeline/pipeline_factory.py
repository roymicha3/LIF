from experiment_manager import Environment, Pipeline
from experiment_manager.common import YAMLSerializable
from experiment_manager.pipelines import PipelineFactory

# Import all pipelines
from pipeline.training_pipeline import TrainingPipeline
from pipeline.sequential_pipeline import SequentialPipeline


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