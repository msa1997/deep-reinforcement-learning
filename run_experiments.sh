#!/bin/bash

# --- Experiment Configurations from Table 1 ---
# Format: m L cl ch cp co issuing_policy
EXPERIMENTS=(
    "2 1 5 1 7 3 LIFO"
    "2 1 5 1 7 3 FIFO"
    "2 1 5 1 10 3 LIFO"
    "2 1 5 1 10 3 FIFO"
    "2 2 5 1 7 3 LIFO"
    "2 2 5 1 7 3 FIFO"
    "2 2 5 1 10 3 LIFO"
    "2 2 5 1 10 3 FIFO"
)

# --- Seeds and Shaping Types ---
SEEDS=(1 2 3)
SHAPING_TYPES=("none" "base_stock" "bsp_low_ew")

# --- Run Experiments ---
for exp_idx in ${!EXPERIMENTS[@]}; do
    params=(${EXPERIMENTS[$exp_idx]})
    m=${params[0]}
    L=${params[1]}
    cl=${params[2]}
    ch=${params[3]}
    cp=${params[4]}
    co=${params[5]}
    policy=${params[6]}

    for seed in ${SEEDS[@]}; do
        for shape_type in ${SHAPING_TYPES[@]}; do
            echo "--- Running Experiment $((exp_idx+1))/8 | Seed $seed/3 | Shaping: $shape_type ---"
            echo "    Params: m=$m, L=$L, cl=$cl, ch=$ch, cp=$cp, co=$co, policy=$policy"
            
            python train.py \
                --m $m \
                --L $L \
                --cl $cl \
                --ch $ch \
                --cp $cp \
                --co $co \
                --issuing_policy $policy \
                --seed $seed \
                --shaping_type $shape_type \
                --log_dir "logs/exp$((exp_idx+1))" \
                --model_dir "models/exp$((exp_idx+1))"
            
            echo "--- Finished Experiment $((exp_idx+1)) | Seed $seed | Shaping: $shape_type ---"
            echo ""
        done
    done
done

echo "All experiments completed!" 