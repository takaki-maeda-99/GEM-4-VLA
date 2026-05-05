"""ABC contract: ChunkPredictor cannot be instantiated, subclasses MUST
override predict / chunk_len / action_dim."""
import numpy as np
import pytest

from vla_project.deployment.predictors.base import ChunkPredictor


def test_chunk_predictor_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        ChunkPredictor()  # type: ignore[abstract]


def test_concrete_subclass_must_override_predict():
    class Bad(ChunkPredictor):
        @property
        def chunk_len(self) -> int: return 1
        @property
        def action_dim(self) -> int: return 1
    with pytest.raises(TypeError, match="abstract"):
        Bad()  # type: ignore[abstract]


def test_concrete_subclass_with_all_overrides_works():
    class Good(ChunkPredictor):
        def predict(self, obs):
            return np.zeros((1, 1), dtype=np.float32)
        @property
        def chunk_len(self) -> int: return 1
        @property
        def action_dim(self) -> int: return 1
    p = Good()
    assert p.chunk_len == 1
    assert p.action_dim == 1
    assert p.predict({}).shape == (1, 1)
