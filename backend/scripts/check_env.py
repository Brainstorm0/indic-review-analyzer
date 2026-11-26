"""Phase-1 requirement check. Run: python -m scripts.check_env"""
import importlib
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
BACKEND = Path(__file__).resolve().parents[1]
def line(name, ok, detail=""):
    print(f"[{'OK' if ok else '!!'}] {name:<30} {detail}")
    return ok
def main():
    all_ok = True
    v = sys.version_info
    all_ok &= line("Python 3.10-3.12", (3, 10) <= (v.major, v.minor) <= (3, 12), platform.python_version())
    line("OS", True, f"{platform.system()} {platform.release()}")
    for pkg in ["torch", "transformers", "sentencepiece", "sklearn", "pandas", "numpy", "scipy",
                "matplotlib", "fastapi", "uvicorn", "openai", "dotenv", "huggingface_hub", "multipart"]:
        try:
            m = importlib.import_module(pkg)
            line(f"package {pkg}", True, getattr(m, "__version__", ""))
        except Exception as e:  # noqa: BLE001
            all_ok &= line(f"package {pkg}", False, f"missing ({e})")
    try:
        import torch

        line("CUDA GPU", True, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "not available (CPU mode is fine)")
    except Exception:  # noqa: BLE001
        pass
    free_gb = shutil.disk_usage(BACKEND).free / 1e9
    all_ok &= line("Free disk >= 6 GB", free_gb >= 6, f"{free_gb:.1f} GB")
    for tool in ("node", "npm"):
        exe = shutil.which(tool)
        ver = ""
        if exe:
            try:
                ver = subprocess.run([exe, "-v"], capture_output=True, text=True, timeout=20).stdout.strip()
            except Exception:  # noqa: BLE001
                ver = "found"
        all_ok &= line(f"{tool} installed", bool(exe), ver)
    line("ollama installed (optional)", bool(shutil.which("ollama")), "")
    env = BACKEND / ".env"
    all_ok &= line(".env present", env.exists(), str(env))
    if env.exists():
        from dotenv import load_dotenv
        load_dotenv(env)
        for k in ("GEMINI_API_KEY", "GROQ_API_KEY"):
            configured = bool(os.getenv(k))
            line(f"{k} set (optional)", configured, "configured" if configured else "empty")
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=8)
        line("huggingface.co reachable", True)
    except Exception as e:  # noqa: BLE001
        all_ok &= line("huggingface.co reachable", False, str(e)[:80])
    print("\nRESULT:", "ALL REQUIRED CHECKS PASSED" if all_ok else "FIX THE [!!] ITEMS ABOVE (see Phase 3)")
    return 0 if all_ok else 1
if __name__ == "__main__":
    raise SystemExit(main())
