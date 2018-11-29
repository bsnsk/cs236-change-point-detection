#!/usr/bin/env bash
python src/run_flow.py   --data_dir "data/raw/" \
                                --iter_max 20000 \
                                --iter_eval 50 \
                                --batch_size 500 \
                                --learning_rate 1e-2 \
                                --reg 0 \
                                --window_size 100 \
                                --hidden_sizes 100 100 100 100
