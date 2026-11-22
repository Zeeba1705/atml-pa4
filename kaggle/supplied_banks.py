import os
import sys
import argparse

import torch
from transformers.generation.utils import GenerationMixin

#setup_data.py (course asset, not edited) passes generator= to model.generate, which transformers 4.57.1 rejects.
#4.57.1 samples with torch.multinomial on the default rng, so we drop the kwarg and seed the default rng with the
#generator's seed. a fresh generator and the default generator seeded with the same value give the same stream.
ORIGINAL_GENERATE= GenerationMixin.generate


def generate_with_generator(self, *args, **kwargs):
    gen= kwargs.pop("generator", None)

    if gen is not None:
        torch.manual_seed(gen.initial_seed())

    return ORIGINAL_GENERATE(self, *args, **kwargs)


def patch_generate():
    GenerationMixin.generate= generate_with_generator


def main(args):
    sys.path.insert(0, os.path.join(args.data_root, "scripts"))
    import setup_data

    patch_generate()

    rev= setup_data.revisions(False)
    setup_data.prefix_banks(rev, args.force, args.limit_prefix_prompts)


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default="atml-pa4-datasets"
    )
    parser.add_argument(
        "--force",
        action="store_true"
    )
    parser.add_argument(
        "--limit_prefix_prompts",
        type=int,
        default=None
    )

    args= parser.parse_args()
    main(args)
