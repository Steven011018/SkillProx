"""LibreOffice discovery and wrapper installation."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def find_soffice() -> str | None:
    """Find the real LibreOffice binary before any wrapper modifies PATH."""
    for candidate in (
        os.environ.get("SE_PIPELINE_REAL_SOFFICE"),
        shutil.which("soffice"),
        shutil.which("libreoffice"),
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    ):
        if candidate and Path(candidate).exists():
            return str(Path(candidate).resolve())
    return None


def install_soffice_wrapper(workdir: Path, lock_path: Path) -> Path | None:
    """Create PATH wrappers so executor-side soffice calls share one lock."""
    real_soffice = find_soffice()
    if not real_soffice:
        return None

    tools_dir = workdir / "_eval_tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    wrapper = '#!/usr/bin/env python3\nimport os\nimport subprocess\nimport sys\nfrom pathlib import Path\n\ntry:\n    import fcntl\nexcept ImportError:\n    fcntl = None\n\nreal = os.environ.get("SE_PIPELINE_REAL_SOFFICE")\nlock_path = os.environ.get("SE_PIPELINE_SOFFICE_LOCK")\nif not real:\n    print("SE_PIPELINE_REAL_SOFFICE is not set", file=sys.stderr)\n    sys.exit(127)\n\ndef run():\n    return subprocess.run([real] + sys.argv[1:]).returncode\n\nif lock_path and fcntl is not None:\n    p = Path(lock_path)\n    p.parent.mkdir(parents=True, exist_ok=True)\n    with open(p, "w") as f:\n        fcntl.flock(f, fcntl.LOCK_EX)\n        try:\n            code = run()\n        finally:\n            fcntl.flock(f, fcntl.LOCK_UN)\nelse:\n    code = run()\nsys.exit(code)\n'
    for name in ("soffice", "libreoffice"):
        path = tools_dir / name
        path.write_text(wrapper, encoding="utf-8")
        path.chmod(0o755)

    os.environ["SE_PIPELINE_REAL_SOFFICE"] = real_soffice
    os.environ["SE_PIPELINE_SOFFICE_LOCK"] = str(lock_path)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if str(tools_dir) not in path_parts:
        os.environ["PATH"] = str(tools_dir) + os.pathsep + os.environ.get("PATH", "")
    return tools_dir
