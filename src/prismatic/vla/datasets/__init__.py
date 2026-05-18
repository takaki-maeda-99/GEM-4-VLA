# GEM-4-VLA vendored slimming: empty so importing
# ``prismatic.vla.datasets.rlds.*`` does not pull in the non-RLDS dataset
# wrappers (DummyDataset / EpisodicRLDSDataset / RLDSBatchTransform /
# RLDSDataset) which transitively need prismatic.models.* backbones (timm).
# v37 only uses the rlds.* submodule directly.
