#!/bin/bash
# 正式 synthetic hierarchy 2000-epoch 调度网格 (DS 接手用)。
# README §6.1: tree ∈ {binary,4ary,8ary,variable} × dim ∈ {2,5,10,20}, nodes 1K/5K/10K, seed=42。
# method: B1..B5 全跑 (anchor_2 已验 baseline 行为)。
#
# 用法:
#   bash run_grid.sh            # 跑全网格 (后台, 每配置一 log)
#   bash run_grid.sh smoke      # 只跑 1 个代表配置 (binary 1K dim10 B5, 50ep) 验证链路
#
# GPU: 自动选空闲卡 (train_formal.pick_gpu)。3 卡可并行, 这里串行提交避免 OOM;
#      如需并行, DS 可给不同配置指定 --device cuda:0/1/2。

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
PY=python3
RUNROOT="runs"
mkdir -p "$RUNROOT" logs

EPOCHS=2000
TREES=(binary 4ary 8ary variable)
DIMS=(2 5 10 20)
NODES=10000            # 代表规模; 完整矩阵可加 1000 5000
METHODS=(B1 B2 B3 B4 B5)

if [ "$1" == "smoke" ]; then
  echo "[smoke] binary 1K dim10 B5 50ep"
  $PY train_formal.py --method B5 --tree binary --nodes 1000 --dim 10 \
      --epochs 50 --out "$RUNROOT/smoke_binary_1k_d10_B5"
  exit 0
fi

for tree in "${TREES[@]}"; do
  for dim in "${DIMS[@]}"; do
    for m in "${METHODS[@]}"; do
      name="${tree}_n${NODES}_d${dim}_${m}"
      out="$RUNROOT/$name"
      log="logs/${name}.log"
      if [ -f "$out/summary.json" ]; then
        echo "[skip] $name (已完成)"; continue
      fi
      echo "[run] $name -> $log"
      nohup $PY train_formal.py --method "$m" --tree "$tree" --nodes "$NODES" \
          --dim "$dim" --epochs "$EPOCHS" --out "$out" > "$log" 2>&1
    done
  done
done
echo "全网格完成。"
