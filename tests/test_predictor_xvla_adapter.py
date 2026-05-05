"""Phase 0: XVLAAdapterChunkPredictor is a typed shell that raises
NotImplementedError on predict(). Phase 1 fills in the SigLIP transform +
tokenize + batch build + Q99 denorm pipeline per spec §Section 5.

This file ensures the constructor signature matches what the spec describes
so the Phase 1 implementer cannot drift the API."""
import pytest

from vla_project.deployment.predictors.xvla_adapter import XVLAAdapterChunkPredictor


def test_construction_takes_documented_args():
    """All ctor args from spec §Section 5 line 466 — signature freeze."""
    p = XVLAAdapterChunkPredictor(
        runtime=None,         # Phase 1 will be ModelRuntime
        tokenizer=None,
        image_transform=None,
        action_q99=None,
        action_chunk_len=8,
        action_dim=7,
        domain_id=0,
    )
    assert p.chunk_len == 8
    assert p.action_dim == 7


def test_predict_raises_not_implemented_in_phase_0():
    p = XVLAAdapterChunkPredictor(
        runtime=None, tokenizer=None, image_transform=None,
        action_q99=None, action_chunk_len=8, action_dim=7, domain_id=0,
    )
    with pytest.raises(NotImplementedError, match="Phase 1"):
        p.predict({})
