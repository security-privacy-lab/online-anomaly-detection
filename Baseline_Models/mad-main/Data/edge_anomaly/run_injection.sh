#!/bin/bash

# Synthetic
python3 densification_events.py --dataset synthetic --num_queries 50 --percentage_events 10 --seed 10
python3 sparsification_events.py --dataset synthetic --num_queries 50 --percentage_events 10 --seed 12

# Hospital
python3 densification_events.py --dataset hospital --num_queries 50 --percentage_events 10 --seed 10
python3 sparsification_events.py --dataset hospital --num_queries 50 --percentage_events 10 --seed 12

# Emails
python3 densification_events.py --dataset emails --num_queries 50 --percentage_events 10 --seed 10
python3 sparsification_events.py --dataset emails --num_queries 50 --percentage_events 10 --seed 12

# Traffic
python3 densification_events.py --dataset traffic --num_queries 50 --percentage_events 10 --seed 10
python3 sparsification_events.py --dataset traffic --num_queries 50 --percentage_events 10 --seed 12
