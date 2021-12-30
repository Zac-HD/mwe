"""
# MWE: minimal working examples, automatically

This is a prototype-grade tool for my own use, combining
delta debugging with automatic inlining of library code.

We'll see how many other evil tricks make it in.


    WARNING: this may remove any safety checks from your code.

    DO NOT USE IT if your code deletes files; or maybe even writes.

"""

import contextlib
import functools
import re
import shutil
import site
import subprocess
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional

import libcst as cst
import libcst.matchers as m
import shed
from libcst.codemod import VisitorBasedCodemodCommand
from importlib.util import find_spec

# We run a list of commands, and check for identical return code(s).
# If stderr contains a Python traceback, also check the exception type(s).
COMMANDS = [
    ###############  YOUR CODE HERE  ###############
    ["python", "repro.py"],
    # [expanduser("~/miniconda3/envs/py38/python.exe"), "repro.py"],
    # [expanduser("~/miniconda3/envs/pypy37/python.exe"), "repro.py"],
    ################################################
]


def get_result(command: list[str], cwd: Path, timeout: timedelta) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout.total_seconds(),
        )
    except subprocess.TimeoutExpired:
        return (-999, ":: TimeoutExpired ::")
    try:
        last_line = result.stderr.splitlines()[-1]
    except IndexError:
        return (result.returncode, "")
    exc_type = ""
    if re.match("[a-zA-Z.]+: ", last_line):
        exc_type = last_line.split(":")[0]
    return (result.returncode, exc_type)


def check(source_code: str, expected: object, timeout: timedelta) -> bool:
    with tempfile.TemporaryDirectory() as td:
        fname = Path(td) / "repro.py"
        fname.write_text(source_code)

        results = [get_result(command, cwd=td, timeout=timeout) for command in COMMANDS]
        return results == expected


def compiles(source_code: str, fname: str = "<string>") -> bool:
    try:
        compile(source_code, fname, "exec")
        return True
    except Exception:
        return False


def delta_debug(fname, expected, timeout):
    """Delta debugging with some Python-specific heuristics.

    Specifically, those are:

        - Integrate the `shed` autoformatter (enforces blank lines etc)
        - Work from end to start to minimize NameErrors
        - Proceed chunk-wise, splitting on two blank lines then one
        - Repeat chunk-wise DD until it stops making progress
        - Then line-wise DD to finish up

    """
    for delimiter in ("\n\n\n", "\n\n", "\n"):
        parts = start_parts = fname.read_text().split(delimiter)
        chunksize = len(parts) // 4

        while parts and chunksize > 0:
            i = len(parts) - chunksize
            print(f"shrinking   {delimiter=} {chunksize=} {len(parts)=}", flush=True)
            while parts and i > 0:
                candidate = delimiter.join(parts[:i] + parts[i + chunksize :])
                if compiles(candidate) and check(candidate, expected, timeout):
                    fname.write_text(candidate)
                    parts = candidate.split(delimiter)
                    print(
                        f"   Shrunk!  {delimiter=} {chunksize=} {i=} {len(parts)=}",
                        flush=True,
                    )
                    continue
                i -= chunksize

            candidate = shed.shed(delimiter.join(parts))
            if check(candidate, expected, timeout):
                fname.write_text(candidate)
                print(f"   Formatted!  {chunksize=} {i=} {len(parts)=}")

            if chunksize <= 5:
                chunksize -= 1
            elif chunksize < 10:
                chunksize = 4
            else:
                chunksize //= 2
        if parts != start_parts and delimiter != "\n":
            delta_debug(fname, expected, timeout)
            break


class RemoveCommentsAndDocstrings(VisitorBasedCodemodCommand):
    DESCRIPTION = "Remove comments and docstrings"

    @classmethod
    def codemod(cls, source_code: str) -> str:
        formatter = cls(cst.codemod.CodemodContext())
        return shed.shed(formatter.transform_module(cst.parse_module(source_code)).code)

    def leave_Comment(self, _, updated_node):
        return cst.RemovalSentinel.REMOVE

    @m.call_if_inside(m.FunctionDef() | m.ClassDef())
    def leave_SimpleString(self, _, updated_node):
        if len(updated_node.quote) == 3:
            return cst.Pass()  # replace docstring with pass
        return updated_node


@functools.lru_cache(maxsize=None)
def should_inline_from(module: str) -> Optional[Path]:
    # We only want to inline local or editable-installed modules
    with contextlib.suppress(Exception):
        fname = Path(find_spec(module).origin)
        if fname.suffix == ".py" and not fname.is_relative_to(*site.PREFIXES):
            return fname
    return None


def mwe(fname: Path) -> None:
    shutil.copy(fname, fname.parent / (fname.name + ".orig"))

    start = time.monotonic()
    expected = [
        get_result(command, cwd=fname.parent, timeout=timedelta(hours=1))
        for command in COMMANDS
    ]
    print(expected)

    elapsed = time.monotonic() - start
    timeout = timedelta(seconds=max(elapsed + 1, elapsed * 2))
    del start, elapsed

    # Start by stripping out any comments and docstrings
    initial = "\n".join(
        line
        for line in fname.read_text().splitlines()
        if not line.lstrip().startswith("#")
    )
    candidate = RemoveCommentsAndDocstrings.codemod(initial)
    if check(candidate, expected, timeout):
        fname.write_text(candidate)

    # TODO: alternate DD with inlining of editable-installed and local imports
    # See https://github.com/tonybaloney/pyinline/blob/master/pyinline/__init__.py
    delta_debug(fname, expected, timeout)


if __name__ == "__main__":
    mwe(Path(__file__).parent / "repro.py")
