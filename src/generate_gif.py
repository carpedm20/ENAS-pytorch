#!/usr/bin/env python

import argparse
from glob import glob

from utils import make_gif

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str)
parser.add_argument("--max_frame", type=int, default=50)
parser.add_argument("--output", type=str, default="sampe.gif")
parser.add_argument("--title", type=str, default="")

if __name__ == "__main__":
    args = parser.parse_args()

    paths = glob(f"./logs/{args.model_name}/networks/*.png")
    make_gif(paths, args.output,
            max_frame=args.max_frame,
            prefix=f"{args.title}\n" if args.title else "")
