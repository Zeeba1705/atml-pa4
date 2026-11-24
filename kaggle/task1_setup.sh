#!/bin/bash
#kaggle task 1 setup, called by kaggle/task1.ipynb after it clones this repo to /tmp/pa4/atml-pa4
set -e
nvidia-smi --query-gpu=index,name,memory.total --format=csv
cd /tmp/pa4
rm -rf atml-pa4-datasets
git clone -q https://github.com/AbDu11aHHH/atml-pa4-datasets.git
pip install -q transformers==4.57.1 huggingface_hub==0.36.2 "datasets>=3,<5" pyyaml

#locate the uploaded data, whatever the dataset slug; kaggle decompresses .gz uploads, so accept .jsonl too
HIT=$(find -L /kaggle/input \( -name 'c4_calibration_locked.jsonl.gz' -o -name 'c4_calibration_locked.jsonl' \) | head -1)
LOCK=$(find -L /kaggle/input -name local_source_revisions.json | head -1)
if [ -z "$HIT" ] || [ -z "$LOCK" ]; then
  echo "DATA NOT FOUND under /kaggle/input: add your dataset via Add Input (8 files incl. local_source_revisions.json)"
  find -L /kaggle/input -maxdepth 4 | head -50
  exit 1
fi
SRC=$(dirname "$HIT")
echo "data from $SRC"
mkdir -p atml-pa4-datasets/data/public_locked
for f in c4_calibration c4_development c4_validation_windows wikitext2_test_windows hellaswag_val gsm8k_train gsm8k_test; do
  dst=atml-pa4-datasets/data/public_locked/${f}_locked.jsonl.gz
  if [ -f "$SRC/${f}_locked.jsonl.gz" ]; then cp "$SRC/${f}_locked.jsonl.gz" "$dst"
  elif [ -f "$SRC/${f}_locked.jsonl" ]; then gzip -c "$SRC/${f}_locked.jsonl" > "$dst"
  else echo "MISSING ${f}_locked.jsonl(.gz) in $SRC"; ls -la "$SRC"; exit 1; fi
done
diff "$LOCK" atml-pa4/splits/data_lock/local_source_revisions.json && echo "LOCK MATCHES"
ls -la atml-pa4-datasets/data/public_locked
echo "code at $(git -C atml-pa4 rev-parse --short HEAD) on $BRANCH"
python -c "import torch, transformers, scipy; print(torch.__version__, transformers.__version__, scipy.__version__)"