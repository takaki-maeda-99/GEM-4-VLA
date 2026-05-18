# GEM-4-VLA vendored slimming: keep this file empty so importing
# ``prismatic.vla.datasets.rlds.*`` does not eagerly evaluate
# ``prismatic.models.*`` (which would require timm + draccus + jsonlines
# + json_numpy + rich + ... — none of which v37's RLDS data path needs).
# Submodule imports keep working via the standard import system, e.g.
# ``from prismatic.models.backbones.vision.siglip_vit import SigLIPViTBackbone``
# still resolves directly. Only the top-level convenience re-exports
# (available_models / load) were lost; v37 does not use them.
