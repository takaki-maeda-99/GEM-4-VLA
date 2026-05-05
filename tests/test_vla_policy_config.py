import pytest

from vla_project.models.vla_policy import VLAPolicyConfig


def test_baseline_projectors_are_single_domain_only() -> None:
    with pytest.raises(ValueError, match="num_domains == 1"):
        VLAPolicyConfig(num_domains=2, use_baseline_projectors=True)


def test_wrist_bridge_and_pool_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        VLAPolicyConfig(num_domains=1, use_wrist_bridge=True, use_wrist_pool=True)


def test_wrist_bridge_requires_available_siglip_layers() -> None:
    with pytest.raises(ValueError, match="layer-mapping"):
        VLAPolicyConfig(num_domains=1, num_blocks=35, use_wrist_bridge=True)


def test_invalid_vision_placeholder_mode_fails_fast() -> None:
    with pytest.raises(ValueError, match="vision_placeholder_mode"):
        VLAPolicyConfig(num_domains=1, vision_placeholder_mode="bad")


def test_action_head_layer_mapping_modes() -> None:
    assert VLAPolicyConfig(
        num_domains=1, num_blocks=4, action_head_layer_mode="first_n"
    ).resolve_action_head_layer_indices() == (1, 2, 3, 4)
    assert VLAPolicyConfig(
        num_domains=1, num_blocks=4, action_head_layer_mode="last_n"
    ).resolve_action_head_layer_indices() == (32, 33, 34, 35)
    assert VLAPolicyConfig(
        num_domains=1, num_blocks=4, action_head_layer_mode="even"
    ).resolve_action_head_layer_indices() == (1, 12, 24, 35)
    assert VLAPolicyConfig(
        num_domains=1,
        num_blocks=4,
        action_head_layer_mode="custom",
        action_head_layer_indices=(2, 8, 16, 35),
    ).resolve_action_head_layer_indices() == (2, 8, 16, 35)


def test_custom_action_head_layer_mapping_validates_length_and_range() -> None:
    with pytest.raises(ValueError, match="length"):
        VLAPolicyConfig(
            num_domains=1,
            num_blocks=4,
            action_head_layer_mode="custom",
            action_head_layer_indices=(1, 2, 3),
        )
    with pytest.raises(ValueError, match=r"\[1, 35\]"):
        VLAPolicyConfig(
            num_domains=1,
            num_blocks=4,
            action_head_layer_mode="custom",
            action_head_layer_indices=(1, 2, 3, 36),
        )


def test_vla_gemma4_baseline_profile_accepts_v25_shape() -> None:
    cfg = VLAPolicyConfig(
        num_domains=1,
        compat_profile="vla_gemma4_baseline",
        num_blocks=24,
        use_baseline_projectors=True,
        use_wrist_bridge=True,
        use_soft_prompt=False,
        freeze_llm_and_aq=True,
        vision_placeholder_mode="unused_range",
    )
    assert cfg.action_head_outputs_actions


def test_vla_gemma4_baseline_profile_rejects_partial_compat() -> None:
    with pytest.raises(ValueError, match="vla_gemma4_baseline"):
        VLAPolicyConfig(
            num_domains=1,
            compat_profile="vla_gemma4_baseline",
            num_blocks=24,
            use_baseline_projectors=True,
            use_wrist_bridge=True,
            use_soft_prompt=True,
            freeze_llm_and_aq=True,
            vision_placeholder_mode="unused_range",
        )


def test_vla_gemma4_baseline_profile_requires_first_n_layers() -> None:
    with pytest.raises(ValueError, match="action_head_layer_mode"):
        VLAPolicyConfig(
            num_domains=1,
            compat_profile="vla_gemma4_baseline",
            num_blocks=24,
            action_head_layer_mode="even",
            use_baseline_projectors=True,
            use_wrist_bridge=True,
            use_soft_prompt=False,
            freeze_llm_and_aq=True,
            vision_placeholder_mode="unused_range",
        )
