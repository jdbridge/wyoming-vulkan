"""Compare the ctypes struct layouts of the ggml backends with what a C compiler makes of the pinned headers.

  python tests/check_abi.py emit <dir>     (in the image: writes layout.c and ctypes.json)
  python tests/check_abi.py compare <dir>  (compares ctypes.json with c.json, the compiled program's output)
Run through tests/check_abi.sh: the bindings must match the libraries in the image.
"""

import ctypes as C
import json
import sys
from pathlib import Path

from wyoming_vulkan.engines import parakeet_cpp, whisper_cpp

# ctypes class -> C type name in the headers
STRUCTS = {
    "parakeet_context_params": (parakeet_cpp.ContextParams, "struct parakeet_context_params"),
    "parakeet_full_params": (parakeet_cpp.FullParams, "struct parakeet_full_params"),
    "whisper_context_params": (whisper_cpp.ContextParams, "struct whisper_context_params"),
    "whisper_full_params": (whisper_cpp.FullParams, "struct whisper_full_params"),
    "whisper_aheads": (whisper_cpp.Aheads, "whisper_aheads"),
    "whisper_vad_params": (whisper_cpp.VadParams, "whisper_vad_params"),
}


def ctypes_layout() -> dict:
    out = {}
    for key, (cls, _) in STRUCTS.items():
        out[key] = {"size": C.sizeof(cls), "fields": {name: getattr(cls, name).offset for name, _ in cls._fields_}}
    return out


def c_source() -> str:
    lines = ['#include <stdio.h>', '#include <stddef.h>', '#include "whisper.h"', '#include "parakeet.h"',
             "int main(void) {", '  printf("{");']
    for i, (key, (cls, ctype)) in enumerate(STRUCTS.items()):
        sep = "," if i else ""
        lines.append(f'  printf("{sep}\\"{key}\\": {{\\"size\\": %zu, \\"fields\\": {{", sizeof({ctype}));')
        for j, (name, _) in enumerate(cls._fields_):
            fsep = "," if j else ""
            lines.append(f'  printf("{fsep}\\"{name}\\": %zu", offsetof({ctype}, {name}));')
        lines.append('  printf("}}");')
    lines += ['  printf("}\\n");', "  return 0;", "}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    mode, folder = sys.argv[1], Path(sys.argv[2])
    if mode == "emit":
        (folder / "layout.c").write_text(c_source())
        (folder / "ctypes.json").write_text(json.dumps(ctypes_layout()))
        return 0
    ours, theirs = json.loads((folder / "ctypes.json").read_text()), json.loads((folder / "c.json").read_text())
    bad = 0
    for key in STRUCTS:
        a, b = ours[key], theirs[key]
        diffs = [f"{n}: ctypes {a['fields'][n]} vs C {b['fields'][n]}" for n in a["fields"] if a["fields"][n] != b["fields"][n]]
        if a["size"] != b["size"]:
            diffs.insert(0, f"size: ctypes {a['size']} vs C {b['size']}")
        print(("MISMATCH " if diffs else "ok       ") + f"{key}: {a['size']} bytes, {len(a['fields'])} fields")
        for d in diffs:
            print("         " + d)
        bad += bool(diffs)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
