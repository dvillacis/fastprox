"""Branches against selects in the 1D TV and projection kernels, from GCC
assembly (AT&T syntax, as on Linux):

    c++ -std=c++17 -O3 -DNDEBUG -Iinclude -Isrc -S src/ops/tv1d.cpp -o tv1d.s
    python3 bench/asm_summary.py tv1d.s [more.s ...]

For every function whose demangled name contains one of the kernel names,
prints its instruction count, conditional jumps, conditional moves or
selects, and min/max instructions. A kernel meant to be branch-free that
shows many conditional jumps has been if-converted back into branches by
the compiler (as GCC did with Condat's bound updates on arm64).
"""

import re
import subprocess
import sys

KERNELS = ("condat", "taut_string", "finite_range", "condat_tau", "prox_row")  # prox_row: inlined kernels
JCC = re.compile(r"^\s+j(?!mp\b)[a-z]+\s")  # x86 conditional jumps
BCC = re.compile(r"^\s+b\.[a-z]+\s|^\s+c[bt]n?z\s|^\s+tbn?z\s")  # arm64 conditional branches
SEL = re.compile(r"^\s+(cmov[a-z]*|f?csel|setcc|set[a-z]+)\s")
MINMAX = re.compile(r"^\s+v?(min|max)[sp][sd]\s|^\s+f(min|max)(nm)?\s")


def demangle(names):
    out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True)
    return dict(zip(names, out.stdout.splitlines())) if out.returncode == 0 else {n: n for n in names}


def functions(path):
    funcs, cur = {}, None
    for line in open(path):
        m = re.match(r"^\s+\.type\s+([^,]+),\s*[@%]function", line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
            continue
        if cur and re.match(rf"^\s+\.size\s+{re.escape(cur)},", line):
            cur = None
            continue
        if cur and re.match(r"^\s+[a-z]", line) and not re.match(r"^\s+\.", line):
            funcs[cur].append(line)
    return funcs


def main():
    for path in sys.argv[1:]:
        funcs = functions(path)
        names = demangle(list(funcs))
        print(f"{path}")
        print(f"  {'instr':>6} {'cond. br':>8} {'select':>6} {'min/max':>7}  function")
        for raw, body in funcs.items():
            name = names.get(raw, raw)
            if not any(k in name for k in KERNELS):
                continue
            jumps = sum(bool(JCC.match(l) or BCC.match(l)) for l in body)
            sels = sum(bool(SEL.match(l)) for l in body)
            mm = sum(bool(MINMAX.match(l)) for l in body)
            short = re.sub(r"\(.*", "", name.replace("(anonymous namespace)::", ""))[:60]
            lam = "weighted" if "solve_weighted" in name else ("uniform" if "prox<" in name else "")
            print(f"  {len(body):6d} {jumps:8d} {sels:6d} {mm:7d}  {short} {lam}")


if __name__ == "__main__":
    main()
