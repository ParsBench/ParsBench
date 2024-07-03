import importlib
import pkgutil

from parsbench.tasks import Task


def load_all_tasks() -> list[Task]:
    """
    Load all tasks from the 'parsbench.tasks' package and return a list of Task objects.

    Returns:
        list[Task]: A list of Task objects representing all tasks found in the 'parsbench.tasks' package.

    """
    tasks: list[Task] = []
    package = "parsbench.tasks"

    for _, module_name, _ in pkgutil.iter_modules([package.replace(".", "/")]):
        module = importlib.import_module(f"{package}.{module_name}")
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, Task) and attr is not Task:
                tasks.append(attr)

    return tasks
