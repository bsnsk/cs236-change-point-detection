#!/usr/bin/env bash
python src/run_autoencoder.py   --data_dir "/Users/yongshangwu/Desktop/Projects/cs236/cs236-change-point-detection/data/raw/" \
                                --iter_max 20000 \
                                --iter_eval 50 \
                                --batch_size 500 \
                                --learning_rate 1e-1 \
                                --reg 0 \
                                --window_size 25 \
                                --hidden_sizes 300 300 \
                                --latent_dim 200