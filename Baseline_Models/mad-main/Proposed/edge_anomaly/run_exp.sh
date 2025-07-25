#!/bin/bash

# Synthetic
python3 main.py --dataset synthetic_densif --window_size 10 --time_init 300 --only_active
python3 score_analyzer.py --dataset synthetic_densif

python3 main.py --dataset synthetic_sparsif --window_size 15 --time_init 300
python3 score_analyzer.py --dataset synthetic_sparsif

# Hospital
python3 main.py --dataset hospital_densif --window_size 20 --time_init 300 --only_active
python3 score_analyzer.py --dataset hospital_densif

python3 main.py --dataset hospital_sparsif --window_size 20 --time_init 300
python3 score_analyzer.py --dataset hospital_sparsif

# Emails
python3 main.py --dataset emails_densif --window_size 3000 --time_init 300 --only_active
python3 score_analyzer.py --dataset emails_densif

python3 main.py --dataset emails_sparsif --window_size 250 --time_init 300
python3 score_analyzer.py --dataset emails_sparsif

# Traffic
python3 main.py --dataset traffic_densif --window_size 20 --time_init 300 --only_active
python3 score_analyzer.py --dataset traffic_densif

python3 main.py --dataset traffic_sparsif --window_size 20 --time_init 300
python3 score_analyzer.py --dataset traffic_sparsif
