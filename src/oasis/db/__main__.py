"""CLI entrypoint for python -m oasis.db [command]."""

import sys

from oasis.db.migrate import main as migrate_main


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] in ("migrate", "up"):
        return migrate_main(args)
    print(f"Unknown command: {args[0]}. Available commands: migrate")
    return 1


if __name__ == "__main__":
    sys.exit(main())
