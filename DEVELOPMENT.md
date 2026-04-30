# Development notes

## Running tests

This project uses pytest via uv:

```bash
PYTHONPATH="" uv run pytest -v
```

The `PYTHONPATH=""` prefix is required when the shell environment has ROS2
on `PYTHONPATH` (e.g. `/opt/ros/humble/...`) — otherwise pytest tries to
load the `launch_testing` plugin and crashes during plugin discovery.
This cannot be fixed via `pyproject.toml` because plugin discovery happens
before the project's pytest config is loaded.
