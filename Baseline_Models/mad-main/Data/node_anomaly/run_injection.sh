#!/bin/bash

# Synthetic
python3 densification_sparsification_events.py --dataset synthetic --num_queries 10 --percentage_events 10 --event_size 3 --seed 10
python3 rewiring_events.py --dataset synthetic --num_queries 10 --percentage_events 10 --event_size 3 --seed 42

# Hospital
python3 densification_sparsification_events.py --dataset hospital --num_queries 10 --percentage_events 10 --event_size 3 --seed 10
python3 rewiring_events.py --dataset hospital --num_queries 10 --percentage_events 10 --event_size 3 --seed 42

# Emails
python3 densification_sparsification_events.py --dataset emails --num_queries 10 --percentage_events 10 --event_size 3 --seed 10
python3 rewiring_events.py --dataset emails --num_queries 10 --percentage_events 10 --event_size 3 --seed 42

# Traffic
python3 densification_sparsification_events.py --dataset traffic --num_queries 10 --percentage_events 10 --event_size 3 --seed 10
python3 rewiring_events.py --dataset traffic --num_queries 10 --percentage_events 10 --event_size 3 --seed 42
