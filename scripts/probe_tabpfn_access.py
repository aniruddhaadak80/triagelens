import inspect
from tabpfn import settings as st
import tabpfn.settings as s

print("=== tabpfn settings object ===")
print([a for a in dir(st) if not a.startswith("__")][:80])

print("\n=== _resolve_model_version ===")
import tabpfn.model_loading as ml
print(inspect.getsource(ml._resolve_model_version))

print("\n=== _get_model_source ===")
print(inspect.getsource(ml._get_model_source)[:2500])