"""Patch numba codegen.py to force PTX ISA version 8.4.

This is needed when the CUDA toolkit in the container is newer than the driver.
Driver 550.144.03 = CUDA 12.4 = PTX 8.4 max.
"""
import sys


def patch():
    path = "/usr/local/lib/python3.12/dist-packages/numba/cuda/codegen.py"
    with open(path) as f:
        content = f.read()

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
        with open(path, "w") as f:
            f.write(new_content)
        print("SUCCESS: Patched numba codegen.py (PTX -> 8.4)")
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
