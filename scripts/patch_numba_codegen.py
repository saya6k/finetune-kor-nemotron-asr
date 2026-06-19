"""Patch numba codegen.py to force PTX ISA version 8.4.

Needed when CUDA toolkit is newer than the GPU driver (e.g. toolkit 12.8 on driver 12.4).
Driver 550.144.03 = CUDA 12.4 = PTX 8.4 max.

Not needed on Blackwell (sm_100+): toolkit/driver versions are matched on DGX Station.
"""
import sys
import sysconfig
from pathlib import Path


def patch():
    # Skip on Blackwell (sm_100+): PTX 9.x works natively, downgrade would break things
    try:
        import torch
        if torch.cuda.is_available():
            sm_major = torch.cuda.get_device_capability(0)[0]
            if sm_major >= 10:
                print("Blackwell sm_100 detected — PTX downgrade not needed. Skipping.")
                return True
    except ImportError:
        pass  # torch not yet installed; proceed with patch as safe fallback

    # Dynamic path: works for any Python version and architecture
    codegen_path = Path(sysconfig.get_paths()["purelib"]) / "numba" / "cuda" / "codegen.py"
    if not codegen_path.exists():
        print(f"WARNING: numba codegen not found at {codegen_path}")
        return False

    with open(codegen_path) as f:
        content = f.read()

    # Idempotent: skip if already patched
    if "_patch_re" in content:
        print(f"Already patched: {codegen_path}")
        return True

    # Anchor: unique string right after ptx decode where we insert the fix
    anchor = "ptx = ptx.decode().strip('\\x00').strip()"

    patch_code = (
        "ptx = ptx.decode().strip('\\x00').strip()\n"
        "        # PATCH: Force PTX ISA version 8.4 for driver 550 compatibility\n"
        "        import re as _patch_re\n"
        "        ptx = _patch_re.sub(r'\\.version\\s+\\d+\\.\\d+', '.version 8.4', ptx)"
    )

    if anchor in content:
        new_content = content.replace(anchor, patch_code)
        with open(codegen_path, "w") as f:
            f.write(new_content)
        print(f"SUCCESS: Patched {codegen_path} (PTX -> 8.4)")
        return True
    else:
        print("FAILED: Anchor not found in codegen.py")
        for i, line in enumerate(content.split("\n")):
            if "ptx.decode" in line:
                print(f"  Line {i}: {repr(line)}")
        return False


if __name__ == "__main__":
    success = patch()
    sys.exit(0 if success else 1)
