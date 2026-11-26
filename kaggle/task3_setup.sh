#!/bin/bash
#kaggle task 3 setup, called by kaggle/task3.ipynb after it clones this repo to /tmp/pa4/atml-pa4
#needs the private dataset with the *_locked.jsonl(.gz) files (same one as task 1); optional inputs: the supplied
#prefix_banks_locked.jsonl(.gz) and the output of an earlier task 3 version (teacher pool, banks, adapters)
set -e
#a session without a gpu (accelerator left at None, or no gpu granted) must stop here with a clear message
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "NO GPU IN THIS SESSION: set Settings -> Accelerator -> GPU T4 x2 (and check no other GPU session is using your quota)"
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv
NGPU=$(nvidia-smi -L | wc -l)
[ "$NGPU" -ge 2 ] || { echo "ONLY $NGPU GPU: task 3 stages use two T4s, pick GPU T4 x2"; exit 1; }
cd /tmp/pa4
rm -rf atml-pa4-datasets
git clone -q https://github.com/AbDu11aHHH/atml-pa4-datasets.git
pip install -q transformers==4.57.1 huggingface_hub==0.36.2 peft==0.21.2 pyyaml
#kaggle ships torchao 0.10; peft 0.21.2 raises ImportError on any torchao < 0.16 when it builds lora layers.
#we never use torchao, so remove it
pip uninstall -y -q torchao || true
python -c "import importlib.util as u; assert u.find_spec('torchao') is None, 'torchao still installed'"

HIT=$(find -L /kaggle/input \( -name 'gsm8k_train_locked.jsonl.gz' -o -name 'gsm8k_train_locked.jsonl' \) | head -1)
LOCK=$(find -L /kaggle/input -name local_source_revisions.json | head -1)
if [ -z "$HIT" ] || [ -z "$LOCK" ]; then
  echo "DATA NOT FOUND under /kaggle/input: add the private dataset (gsm8k_*_locked + local_source_revisions.json)"
  find -L /kaggle/input -maxdepth 4 | head -50
  exit 1
fi
SRC=$(dirname "$HIT")
echo "data from $SRC"
mkdir -p atml-pa4-datasets/data/public_locked
for f in gsm8k_train gsm8k_test; do
  dst=atml-pa4-datasets/data/public_locked/${f}_locked.jsonl.gz
  if [ -f "$SRC/${f}_locked.jsonl.gz" ]; then cp "$SRC/${f}_locked.jsonl.gz" "$dst"
  elif [ -f "$SRC/${f}_locked.jsonl" ]; then gzip -c "$SRC/${f}_locked.jsonl" > "$dst"
  else echo "MISSING ${f}_locked.jsonl(.gz) in $SRC"; ls -la "$SRC"; exit 1; fi
done
diff "$LOCK" atml-pa4/splits/data_lock/local_source_revisions.json && echo "LOCK MATCHES"

#supplied prefix banks (needed from step 5 on)
BANK=$(find -L /kaggle/input \( -name 'prefix_banks_locked.jsonl.gz' -o -name 'prefix_banks_locked.jsonl' \) | head -1)
mkdir -p atml-pa4-datasets/data/prefix_banks
if [ -n "$BANK" ]; then
  case "$BANK" in *.gz) cp "$BANK" atml-pa4-datasets/data/prefix_banks/ ;; *) gzip -c "$BANK" > atml-pa4-datasets/data/prefix_banks/prefix_banks_locked.jsonl.gz ;; esac
  echo "supplied prefix banks from $BANK"
else
  echo "supplied prefix banks not attached (fine for stage 1)"
fi

#outputs of an earlier task 3 version, if attached as input: copy into /kaggle/working so finished items are skipped
PREV=$(find -L /kaggle/input -path '*outputs/task3' -type d | head -1)
if [ -n "$PREV" ]; then
  mkdir -p /kaggle/working/outputs && cp -rn "$PREV" /kaggle/working/outputs/ && echo "previous task 3 outputs from $PREV"
fi
PREVRES=$(find -L /kaggle/input -path '*results/task3' -type d | head -1)
if [ -n "$PREVRES" ]; then
  mkdir -p /kaggle/working/results && cp -rn "$PREVRES" /kaggle/working/results/ && echo "previous task 3 results from $PREVRES"
fi

echo "code at $(git -C atml-pa4 rev-parse --short HEAD) on $BRANCH"
python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)"
