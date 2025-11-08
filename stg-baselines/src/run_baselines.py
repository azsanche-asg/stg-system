import argparse

from src.bsp_depth import run_bsp_depth
from src.scenegraph_lite import run_scenegraph
from src.seg_repeat import run_seg_repeat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["bsp", "repeat", "graph"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.baseline == "bsp":
        run_bsp_depth(args.input, args.output)
    elif args.baseline == "repeat":
        run_seg_repeat(args.input, args.output)
    else:
        run_scenegraph(args.input, args.output)


if __name__ == "__main__":
    main()
