#!/bin/bash

# We send the anomalous datasets, queries file and label files to their respective algorithms folders

datasets=("synthetic" "hospital" "emails" "traffic")
anomalies=("densif_sparsif" "rewiring")

for dat in ${datasets[@]}; do
	for anom in "${anomalies[@]}"; do
		echo Processing ${dat}_${anom}
		# dataset
		cp ./anomalous_data/${dat}_${anom}_data.txt ../../Baselines/AnomRank_MinjiYoon_2019/datasets
		cp ./anomalous_data/${dat}_${anom}_data.txt ../../Baselines/LAD_ShenyangHuang_2020/datasets
		cp ./anomalous_data/${dat}_${anom}_data.txt ../../Proposed/graph_anomaly/datasets
		# labels
		cp ./anomalous_data/${dat}_${anom}_gt.txt ../../Baselines/AnomRank_MinjiYoon_2019/labels
		cp ./anomalous_data/${dat}_${anom}_gt.txt ../../Baselines/LAD_ShenyangHuang_2020/labels
		cp ./anomalous_data/${dat}_${anom}_gt.txt ../../Proposed/graph_anomaly/labels
	done
done
