#!/bin/bash
echo Copy data
python split_data.py
echo Preprocess data
python prepare_data_inst.py --data_split train
python prepare_data_inst.py --data_split val
python prepare_data_inst.py --data_split test
python prepare_data_inst_gttxt.py
