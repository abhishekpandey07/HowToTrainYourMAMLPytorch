#!/bin/sh
cd ../..
export DATASET_DIR="datasets/"
# Activate the relevant virtual environment:

python train_maml_system.py --name_of_args_json_file experiment_config/twitter_config/twitter_maml++_twitter_5_way_10_shot.json --gpu_to_use 0