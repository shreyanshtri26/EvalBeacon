"""Run a command and assert its exit code, explaining what a mismatch means. Used by CI.

  python scripts/expect_exit.py 1 -- uv run evalbeacon demo --candidate candidate-bad

EvalBeacon exit codes: 0 pass/stable, 1 gate failed/drift, 2 setup or provider error, 3 inconclusive.
Exit 0 here means the command exited with the expected code. Anything else exits 1 with a diagnosis, so a
setup/provider failure (2) is never mistaken for the quality regression the step expected (1), and an
unexpected pass is reported as such.
"""
import subprocess
import sys

MEANING = {0: "pass/stable", 1: "gate failed or drift detected", 2: "SETUP OR PROVIDER ERROR (not a quality result)",
           3: "inconclusive (too few cases)"}


def main(argv: list[str]) -> int:
    if len(argv) < 4 or argv[1] != "--" and argv[2] != "--":
        print("usage: expect_exit.py EXPECTED_CODE -- COMMAND [ARGS...]", file=sys.stderr)
        return 64
    try:
        expected = int(argv[0])
    except ValueError:
        print(f"expected exit code must be an integer, got {argv[0]!r}", file=sys.stderr)
        return 64
    cmd = argv[argv.index("--") + 1:]
    if not cmd:
        print("no command given after --", file=sys.stderr)
        return 64
    actual = subprocess.run(cmd).returncode
    if actual == expected:
        print(f"expect_exit: OK, exit {actual} ({MEANING.get(actual, 'unknown')}) as expected")
        return 0
    print(f"expect_exit: FAIL, expected exit {expected} ({MEANING.get(expected, 'unknown')}) but got {actual} "
          f"({MEANING.get(actual, 'command failed or crashed')})", file=sys.stderr)
    if actual == 2:
        print("expect_exit: exit 2 means the run could not be evaluated (bad key, broken baseline, bad policy, missing file); "
              "fix the setup instead of treating this as a regression", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
