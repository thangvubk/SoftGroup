#!/bin/bash
python prepare_data_inst.py
python downsample.py
python prepare_data_inst_gttxt.py
