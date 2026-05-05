"""Smoke: deployment package imports without errors. Will be replaced by
test_deployment_schemas.py in Task 2."""

def test_deployment_package_imports():
    import vla_project.deployment  # noqa: F401
    import vla_project.deployment.predictors  # noqa: F401
