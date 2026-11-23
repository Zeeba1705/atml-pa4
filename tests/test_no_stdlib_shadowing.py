import os
import sys

ROOT= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIRS= ["common", "task1", "task2", "task3", "kaggle", "tests"]


def test_no_module_shadows_stdlib():
    #running a script by path puts its folder first on sys.path, so e.g. kaggle/queue.py breaks "import torch"
    clashes= []

    for folder in SCRIPT_DIRS:
        for name in os.listdir(os.path.join(ROOT, folder)):
            module= name[:-3] if name.endswith(".py") else None

            if module and module != "__init__" and module in sys.stdlib_module_names:
                clashes.append(f"{folder}/{name}")

    assert clashes == []
