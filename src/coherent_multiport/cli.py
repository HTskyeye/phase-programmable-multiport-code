"""Small command-line interface; cloud submissions are explicit."""

import argparse
import logging
from pathlib import Path

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Coherent multiport inverse design")
    parser.add_argument("command", choices=("preview", "optimize", "evaluate"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Default: results/<device>")
    parser.add_argument(
        "--submit", action="store_true", help="Allow paid Tidy3D cloud jobs"
    )
    parser.add_argument(
        "--max-steps", type=int, help="Maximum additional updates in this invocation"
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.command != "preview" and not args.submit:
        parser.error("Cloud work requires --submit")
    if args.command != "optimize" and args.max_steps is not None:
        parser.error("--max-steps applies only to optimize")
    directory = args.output or Path("results") / cfg.device
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "preview":
        from .postprocess import preview

        preview(cfg, directory)
    elif args.command == "optimize":
        from .optimization import optimize

        optimize(cfg, directory, submit=args.submit, max_steps=args.max_steps)
    else:
        from .postprocess import evaluate

        evaluate(cfg, directory, submit=args.submit)


if __name__ == "__main__":
    main()
