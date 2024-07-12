from pathlib import Path


def get_task_path(
    path: str, model_name: str, task_name: str, make_dirs: bool = True
) -> Path:
    model_name = model_name.replace("/", "_").replace("-", "_")
    task_name = task_name.replace(" ", "_").replace("-", "_")

    task_path = Path(path) / model_name / task_name
    if not task_path.exists() and make_dirs:
        task_path.mkdir(parents=True)

    return task_path


def check_task_matches_exists(
    task_path: Path, n_shots: int, sub_task: str | None = None
) -> bool:
    if sub_task:
        matches_path = task_path / f"matches_{sub_task}_{n_shots}_shot.jsonl"
    else:
        matches_path = task_path / f"matches_{n_shots}_shot.jsonl"
    return matches_path.exists()
