# GEM-4-VLA vendored slimming: same rationale as prismatic/__init__.py.
# Empty top-level keeps ``import prismatic.vla.datasets.rlds.dataset`` from
# pulling in ``prismatic.vla.materialize``, which in turn imports model
# backbones (timm). v37 only needs the data-pipeline submodules.
