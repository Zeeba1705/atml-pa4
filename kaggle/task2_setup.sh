#!/bin/bash
#kaggle task 2 setup, called by kaggle/task2.ipynb after it clones this repo to /tmp/pa4/atml-pa4
#mqar data are in the course data repo itself (no kaggle dataset needed)
set -e
nvidia-smi --query-gpu=index,name,memory.total --format=csv
cd /tmp/pa4
rm -rf atml-pa4-datasets
git clone -q https://github.com/AbDu11aHHH/atml-pa4-datasets.git
pip install -q transformers==4.57.1 huggingface_hub==0.36.2 pyyaml
ls -la atml-pa4-datasets/data/mqar
echo "code at $(git -C atml-pa4 rev-parse --short HEAD) on $BRANCH, data repo at $(git -C atml-pa4-datasets rev-parse --short HEAD)"
nproc
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
