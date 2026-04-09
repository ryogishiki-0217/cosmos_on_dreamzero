import sys
import argparse
from pathlib import Path

import tomllib
from packaging.requirements import Requirement
from importlib import metadata


def load_requirements(pyproject_path: Path):
    data = tomllib.loads(pyproject_path.read_text())
    project = data.get("project", {})
    raw_reqs = project.get("dependencies", [])
    reqs: list[Requirement] = []
    for raw in raw_reqs:
        try:
            req = Requirement(raw)
        except Exception as exc:  # fallback in case of malformed entries
            print(f"[WARN] Skip unparsable requirement: {raw} ({exc})")
            continue
        if req.marker and not req.marker.evaluate():
            continue
        reqs.append(req)
    return reqs


def find_site_packages(venv_path: Path):
    """Return candidate site-packages directories under a venv."""
    candidates = []
    for lib_dir in ("lib", "lib64"):
        py_dir = next(py for py in ("python3.12", "python3.11", "python3.10") if (venv_path / lib_dir / py).exists()) if any(
            (venv_path / lib_dir / py).exists() for py in ("python3.12", "python3.11", "python3.10")
        ) else None
        if py_dir:
            sp = venv_path / lib_dir / py_dir / "site-packages"
            if sp.exists():
                candidates.append(sp)
    return candidates


def build_installed_index(paths=None):
    """Build a map of installed distributions name -> version, optionally scoped to paths."""
    installed = {}
    for dist in metadata.distributions(path=paths):
        installed[dist.metadata["Name"].lower()] = dist.version
    return installed


def check_requirements(reqs: list[Requirement], installed_index: dict[str, str]):
    results = {
        "total": len(reqs),
        "ok": 0,
        "missing": [],
        "mismatch": [],
    }

    for req in reqs:
        name = req.name
        key = name.lower()
        installed_version = installed_index.get(key)

        if installed_version is None:
            results["missing"].append((name, str(req.specifier) or "(any)"))
            continue

        # If no specifier, any version is acceptable
        if not req.specifier:
            results["ok"] += 1
            continue

        if req.specifier.contains(installed_version, prereleases=True):
            results["ok"] += 1
        else:
            results["mismatch"].append(
                (
                    name,
                    installed_version,
                    str(req.specifier),
                )
            )

    return results


def print_report(results):
    total = results["total"]
    ok = results["ok"]
    missing = results["missing"]
    mismatch = results["mismatch"]

    print("== DreamZero dependency check ==")
    print(f"Total requirements: {total}")
    print(f"Satisfied:         {ok}")
    print(f"Missing:           {len(missing)}")
    print(f"Version mismatch:  {len(mismatch)}")
    print(f"Missing packages list:   {', '.join(name for name, _ in missing) if missing else 'none'}")
    print(
        f"Mismatch packages list:  {', '.join(name for name, _, _ in mismatch) if mismatch else 'none'}"
    )
    print()

    if missing:
        print("-- Missing packages --")
        for name, spec in missing:
            print(f"  {name}  required: {spec}")
        print()

    if mismatch:
        print("-- Version mismatches --")
        for name, installed, spec in mismatch:
            print(f"  {name}  installed: {installed}  required: {spec}")
        print()

    if not missing and not mismatch:
        print("All dependencies satisfy the specified versions.")


def main():
    parser = argparse.ArgumentParser(description="Check dreamzero dependencies using importlib.metadata")
    parser.add_argument(
        "--pyproject",
        default="/home/lingsheng/chennuo/dreamzero/pyproject.toml",
        help="Path to pyproject.toml",
    )
    parser.add_argument(
        "--venv",
        default="/home/lingsheng/chennuo/cosmos-policy/.venv",
        help="Path to the venv to inspect (site-packages will be auto-detected)",
    )
    args = parser.parse_args()

    pyproject_path = Path(args.pyproject)
    if not pyproject_path.exists():
        print(f"pyproject.toml not found at {pyproject_path}")
        sys.exit(1)

    venv_path = Path(args.venv)
    site_packages_paths = find_site_packages(venv_path) if venv_path.exists() else None
    if site_packages_paths:
        installed_index = build_installed_index(paths=[str(p) for p in site_packages_paths])
    else:
        installed_index = build_installed_index()

    reqs = load_requirements(pyproject_path)
    results = check_requirements(reqs, installed_index)
    print_report(results)


if __name__ == "__main__":
    main()
