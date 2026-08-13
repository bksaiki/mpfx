#!/usr/bin/env python3
"""Dump the generated code around the FP control/status register accesses.

  asm_dump.py --dump FILE.s              # window on an existing listing
  asm_dump.py --tu SOURCE.cpp [DB]       # recompile a TU to assembly, then dump

--tu reuses the exact command CMake used, so the listing matches the shipped
object file rather than an approximation of it.
"""

import json
import os
import re
import subprocess
import sys

PATTERN = re.compile(r"mxcsr|fpcr|fpsr", re.IGNORECASE)


def dump(path, context=10, limit=120):
    with open(path) as f:
        lines = f.read().splitlines()
    hits = [i for i, l in enumerate(lines) if PATTERN.search(l)]
    if not hits:
        print(f"  (no fpcr/fpsr/mxcsr access found in {path})")
        return
    shown, printed = set(), 0
    for h in hits:
        for i in range(max(0, h - context), min(len(lines), h + context + 1)):
            if i in shown:
                continue
            shown.add(i)
            mark = ">>" if PATTERN.search(lines[i]) else "  "
            print(f"  {mark} {i + 1:6d}  {lines[i]}")
            printed += 1
            if printed >= limit:
                print(f"  ... truncated at {limit} lines")
                return
        print("  --")


def compile_tu(source, db_path):
    with open(db_path) as f:
        db = json.load(f)
    entry = next((e for e in db if e["file"].endswith(source)), None)
    if entry is None:
        sys.exit(f"no compile command for {source} in {db_path}")
    cmd = entry.get("command") or " ".join(entry["arguments"])
    out = os.path.join(entry["directory"], "asm_dump.s")
    cmd = re.sub(r"-o\s+\S+", "-o " + out, cmd)
    cmd = cmd.replace(" -c ", " -S ")
    print(f"  $ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=entry["directory"],
                       stderr=subprocess.DEVNULL)
    if r.returncode != 0:
        sys.exit(f"compile failed ({r.returncode})")
    return out


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--dump":
        dump(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "--tu":
        db = sys.argv[3] if len(sys.argv) > 3 else "build/compile_commands.json"
        dump(compile_tu(sys.argv[2], db))
    else:
        sys.exit(__doc__)
