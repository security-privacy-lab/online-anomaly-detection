#!/bin/bash

# Synthetic
python3 main.py --dataset synthetic_densif_sparsif --window_size 10 --time_init 300
python3 score_analyzer.py --dataset synthetic_densif_sparsif

python3 main.py --dataset synthetic_rewiring --window_size 10 --time_init 300
python3 score_analyzer.py --dataset synthetic_rewiring 

# Hospital
python3 main.py --dataset hospital_densif_sparsif --window_size 10 --time_init 300
python3 score_analyzer.py --dataset hospital_densif_sparsif

python3 main.py --dataset hospital_rewiring --window_size 10 --time_init 300
python3 score_analyzer.py --dataset hospital_rewiring 

# Emails
python3 main.py --dataset emails_densif_sparsif --window_size 10 --time_init 300
python3 score_analyzer.py --dataset emails_densif_sparsif

python3 main.py --dataset emails_rewiring --window_size 10 --time_init 300
python3 score_analyzer.py --dataset emails_rewiring 

# Traffic
python3 main.py --dataset traffic_densif_sparsif --window_size 10 --time_init 300
python3 score_analyzer.py --dataset traffic_densif_sparsif

python3 main.py --dataset traffic_rewiring --window_size 10 --time_init 300
python3 score_analyzer.py --dataset traffic_rewiring
