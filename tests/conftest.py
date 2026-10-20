import importlib
import os
import pkgutil
import sys
from pathlib import Path

# With DISCRIMALIGN_TEST_INSTALLED=1 the tests run against the installed
# discrimalign package instead of the checkout's src/: every `src.<module>` the
# tests import is the installed `discrimalign.<module>`. CI uses this to test
# the built wheel.
if os.environ.get("DISCRIMALIGN_TEST_INSTALLED") == "1":
    assert "src" not in sys.modules, "src was imported before the alias was set"
    import discrimalign

    repo = Path(__file__).resolve().parents[1]
    package_dir = Path(discrimalign.__file__).resolve().parent
    assert repo not in package_dir.parents, f"discrimalign imported from the checkout: {package_dir}"

    sys.modules["src"] = discrimalign
    for info in pkgutil.iter_modules(discrimalign.__path__):
        sys.modules[f"src.{info.name}"] = importlib.import_module(f"discrimalign.{info.name}")
