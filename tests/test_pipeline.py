from pydantic_ai.models.test import TestModel
from src.pattern_pipeline import extractor, outliner, drafter, pipeline_run


def test_pipeline_without_network():
    # Override models with TestModel to avoid network calls
    with (
        extractor.override(model=TestModel()),
        outliner.override(model=TestModel()),
        drafter.override(model=TestModel()),
    ):
        draft = pipeline_run("Write about unit testing for data pipelines.")
        assert isinstance(draft, str)
        assert len(draft) > 0
