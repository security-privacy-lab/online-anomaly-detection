#!/bin/bash

# We send the anomalous datasets, queries file and label files to their respective algorithms folders

datasets=("synthetic" "hospital" "emails" "traffic")
anomalies=("densif" "densif_sparsif" "rewiring")

for dat in ${datasets[@]}; do
	for anom in "${anomalies[@]}"; do
		echo Processing ${dat}_${anom}
		# dataset
		cp ./anomalous_data/${dat}_${anom}_data.txt ../../Baselines/FFADE-N_YenChang_2021/datasets
		cp ./anomalous_data/${dat}_${anom}_data.txt ../../Baselines/DynAnom_XingzhiGuo_2022/datasets
		cp ./anomalous_data/${dat}_${anom}_data.txt ../../Proposed/node_anomaly/datasets
		# queries
		cp ./anomalous_data/${dat}_${anom}_queries.txt ../../Baselines/FFADE-N_YenChang_2021/queries
		cp ./anomalous_data/${dat}_${anom}_queries.txt ../../Baselines/DynAnom_XingzhiGuo_2022/queries
		cp ./anomalous_data/${dat}_${anom}_queries.txt ../../Proposed/node_anomaly/queries
		# labels
		cp ./anomalous_data/${dat}_${anom}_gt.txt ../../Baselines/FFADE-N_YenChang_2021/labels
		cp ./anomalous_data/${dat}_${anom}_gt.txt ../../Baselines/DynAnom_XingzhiGuo_2022/labels
		cp ./anomalous_data/${dat}_${anom}_gt.txt ../../Proposed/node_anomaly/labels
	done
done
