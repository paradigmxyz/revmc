#!/usr/bin/env python3
"""Annotate an LLVM IR dump with source lines from bytecode.txt.

Usage: annotate_ir.py <ir_file> <bytecode_file> [output_file]

Resolves !dbg references to !DILocation metadata and appends the
corresponding bytecode.txt source line as a comment.
"""

import re
import sys


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <ir_file> <bytecode_file> [output_file]", file=sys.stderr)
        sys.exit(1)

    ir_path = sys.argv[1]
    src_path = sys.argv[2]
    out = open(sys.argv[3], "w") if len(sys.argv) > 3 else sys.stdout

    with open(src_path) as f:
        src_lines = f.read().splitlines()

    with open(ir_path) as f:
        ir_text = f.read()

    # Parse !N = !DILocation(line: L, ...) metadata.
    di_loc = {}
    for m in re.finditer(r"^!(\d+)\s*=\s*!DILocation\(line:\s*(\d+)", ir_text, re.MULTILINE):
        di_loc[int(m.group(1))] = int(m.group(2))

    # Annotate each IR line that has a !dbg !N reference.
    dbg_re = re.compile(r"!dbg !(\d+)")
    for line in ir_text.splitlines():
        m = dbg_re.search(line)
        if m:
            meta_id = int(m.group(1))
            src_line = di_loc.get(meta_id)
            if src_line and 1 <= src_line <= len(src_lines):
                text = src_lines[src_line - 1].strip()
                if text:
                    line = f"{line}  ; >> {text}"
        print(line, file=out)

    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    main()
