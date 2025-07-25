#!/bin/bash

# Synthetic
python3 densification_sparsification_events.py --dataset synthetic --percentage_events 1 --event_size 5 --seed 10
python3 rewiring_events.py --dataset synthetic --percentage_events 1 --event_size 5 --seed 42

# Hospital
python3 densification_sparsification_events.py --dataset hospital --percentage_events 1 --event_size 5 --seed 10
python3 rewiring_events.py --dataset hospital --percentage_events 1 --event_size 5 --seed 42

# Emails
python3 densification_sparsification_events.py --dataset emails --percentage_events 1 --event_size 5 --seed 10
python3 rewiring_events.py --dataset emails --percentage_events 1 --event_size 5 --seed 42

# Traffic
python3 densification_sparsification_events.py --dataset traffic --percentage_events 1 --event_size 5 --seed 10
python3 rewiring_events.py --dataset traffic --percentage_events 1 --event_size 5 --seed 42
