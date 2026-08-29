#!/bin/bash
# One-command environment setup for a fresh pod (everything durable lives on
# /workspace; this only rebuilds the pod-local pip env and HF cache).
set -euo pipefail
pip install -q torch transformers accelerate datasets huggingface_hub \
    math-verify scipy pandas matplotlib
pip install -q -e /workspace/jacobian-lens
pip install -q -e /workspace/jlens-cot
python3 - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-4B')
EOF
python3 - <<'EOF'
from extcot.data import load_problems, load_mmlu, load_sst2
load_problems(); load_mmlu(); load_sst2()
print("datasets cached")
EOF
python3 -c "import torch; print('cuda ok:', torch.cuda.get_device_name(0))"
echo "setup done"
