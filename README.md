# EvA
Evolutionary Architectures

For the domain example Human Activity Recoginition (HAR), this programm, will take as input your HAR data and will find a good deep learning architecture, pipeline for it.
It will return you a model.tflite (including preprocessing layers) that you can use in your application.

# How to
- src/experiments holds all executable files
- in src/runner.py, import the experiment file, you want to run
- all programm executions with python3 src/runner.py

## Setup
- conda environment required
- create data folder in root, create dataset folder, insert your data.csv
- make sure the data path is loaded in the experiment.py file that you run
- add a config.py in the src folder, or comment out send_telegram
    telegram_chat_id = ...
    telegram_bot_key = ...

## Info
- when executing the logging func will create a log.txt file in root, which is gitignored
- the experiemtents will create the gitignored folder src/saved_experiments, each experiment creates his subfolder (datetime-experiment_name) in it with all experiement result files in it
