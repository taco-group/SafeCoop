python - <<'PY'
import sys, inspect, glob
import carla
print("carla file:", inspect.getfile(carla))

# Try to get a pip/egg distribution if available
try:
    import pkg_resources as pr
    dist = pr.get_distribution("carla")
    print("carla dist:", dist, "at", dist.location)
except Exception as e:
    print("pkg_resources couldn't find a dist:", e)

# List all carla eggs seen on sys.path (to catch duplicates/mismatches)
hits = [p for p in sys.path if "carla" in p.lower()]
eggs = [p for p in hits if p.endswith(".egg") or "carla-" in p]
print("sys.path hits:", hits)
print("egg candidates:", eggs)
PY