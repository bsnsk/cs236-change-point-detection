#!/usr/bin/env bash
python src/run_flow.py          --data="eeg" \
                                --data_dir "data/raw/" \
                                --iter_max 20000 \
                                --iter_eval 50 \
                                --batch_size 500 \
                                --learning_rate 1e-2 \
                                --reg 1e-3 \
                                --window_size 30 \
                                --hidden_sizes 500 500 500 500

# python src/run_flow.py   --data_dir "data/raw/" \
#                                 --iter_max 20000 \
#                                 --iter_eval 50 \
#                                 --batch_size 500 \
#                                 --learning_rate 1e-2 \
#                                 --reg 1e-3 \
#                                 --window_size 100 \
#                                 --hidden_sizes 100 100 100 100
