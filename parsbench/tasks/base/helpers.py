from pathlib import Path


def get_task_path(path: str, model_name: str, task_name: str) -> Path:
    model_name = model_name.replace("/", "_").replace("-", "_")
    task_name = task_name.replace(" ", "_").replace("-", "_")

    task_path = Path(path) / model_name / task_name
    if not task_path.exists():
        task_path.mkdir(parents=True)

    return task_path
