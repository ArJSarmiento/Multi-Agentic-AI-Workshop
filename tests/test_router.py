from pydantic_ai.models.test import TestModel
from src.pattern_router import router


def test_router_runs_locally():
    with router.override(model=TestModel()):
        r = router.run_sync("What is 2+2?")
        assert r.output is not None
