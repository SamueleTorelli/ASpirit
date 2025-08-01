import os
import sys

from glob      import glob
from argparse  import ArgumentParser
from functools import reduce

import tables as tb


def is_leaf(node):
    return not isinstance(node, tb.Group)


def is_indexed_table(node):
    return isinstance(node, tb.Table) and node.indexed


parser = ArgumentParser()
parser.add_argument("-i", "--input-folder", type=os.path.abspath)
parser.add_argument("-o", "--output-file" , type=os.path.abspath, default="merged.h5")
parser.add_argument("-e", "--extension"   , type=str            , default="h5")
parser.add_argument("--overwrite"   , action="store_true")
parser.add_argument("--ignore-empty", action="store_true")

args = parser.parse_args(sys.argv[1:])
first, *rest = sorted(glob(os.path.join(args.input_folder, f"*.{args.extension}")))

tb.copy_file(first, args.output_file, overwrite=args.overwrite)

to_index = {}
with tb.open_file(args.output_file, "a") as h5out:
    for filename in rest:
        try:
            tb.open_file(filename).close()
        except:
            print(f"Could not open {filename}")
            continue
        with tb.open_file(filename) as h5in:
            for node_out in filter(is_leaf, h5out.walk_nodes()):
                path_tokens = node_out._v_pathname.split("/")[1:]
                try:
                    node_in = reduce(getattr, path_tokens, h5in.root)
                except tb.exceptions.NoSuchNodeError:
                    if   args.ignore_empty: continue
                    else                  : raise RuntimeError(f"Could not read node {'/'.join(path_tokens)} from file {filename}")

                node_out.append(node_in[:])

                if is_indexed_table(node_in):
                    to_index[node_out] = node_in.indexedcolpathnames

    for node, columns in to_index.items():
        for column in columns:
            node.colinstances[column].create_index()

    h5out.flush() # crashes without explicitly flushing
