set -eu
for node in watgpu408 watgpu1103 watgpu1208 watgpu1109; do
  scontrol show node "$node" | grep -E 'NodeName=|CPUAlloc=|Gres=|RealMemory=|State=|AllocTRES='
done
ssh -o BatchMode=yes -o ConnectTimeout=8 watgpu408 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
