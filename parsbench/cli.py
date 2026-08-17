"""ParsBench CLI. `parsbench test [pytest args]` runs your eval tests;
`parsbench view` opens the local evaluation viewer."""

import argparse
import importlib.util
import subprocess
import sys


def _pytest_available() -> bool:
    return importlib.util.find_spec("pytest") is not None


def main(argv=None):
    parser = argparse.ArgumentParser(prog="parsbench")
    sub = parser.add_subparsers(dest="command", required=True)
    test = sub.add_parser("test", help="run eval tests via pytest")
    test.add_argument("pytest_args", nargs="*", help="passed through to pytest")
    view = sub.add_parser("view", help="open the local evaluation viewer")
    view.add_argument("path", nargs="?", default=".parsbench",
                      help="run store, or a directory containing .parsbench")
    view.add_argument("--port", type=int, default=1404,
                      help="port to serve on (walks upward if busy)")
    view.add_argument("--host", default="127.0.0.1")
    view.add_argument("--no-open", action="store_true",
                      help="don't open the browser")
    ns = parser.parse_args(argv)

    if ns.command == "test":
        if not _pytest_available():
            print("parsbench test needs pytest: pip install pytest "
                  "(or parsbench[test])", file=sys.stderr)
            return 1
        # no args → pytest's own discovery, exactly like running `pytest`
        return subprocess.call([sys.executable, "-m", "pytest", *ns.pytest_args])

    if ns.command == "view":
        import parsbench.ui.server

        return parsbench.ui.server.serve(
            ns.path, host=ns.host, port=ns.port, open_browser=not ns.no_open
        )


if __name__ == "__main__":
    sys.exit(main())
