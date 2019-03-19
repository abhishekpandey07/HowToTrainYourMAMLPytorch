#!/bin/sh
cd ../..
export DATASET_DIR="datasets/"
export JSON_CONFIG="experiment_config/twitter_config/twitter_maml++_twitter_5_way_10_shot.json"
# Activate the relevant virtual environment:

python train_maml_system.py --name_of_args_json_file  $JSON_CONFIG >> twitter_5_way_10_shot.log 2>&1
