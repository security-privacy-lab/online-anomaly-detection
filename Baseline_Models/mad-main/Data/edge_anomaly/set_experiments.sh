#!/bin/bash

# We send the anomalous datasets, queries file and label files to their respective algorithms folders

datasets=("synthetic" "hospital" "emails" "traffic")
anomalies=("densif" "sparsif")

for dat in ${datasets[@]}; do
	for anom in "${anomalies[@]}"; do
		echo Processing ${dat}_${anom}
		# dataset
		cp ./anomalous_data/${dat}_${anom}_data.txt ../../Baselines/FFADE_YenChang_2021/datasets
		cp ./anomalous_data/${dat}_${anom}_data.txt ../../Baselines/MIDAS_SiddharthBhatia_2020/datasets
		cp ./anomalous_data/${dat}_${anom}_data.txt ../../Proposed/edge_anomaly/datasets
		# queries
		cp ./anomalous_data/${dat}_${anom}_queries.txt ../../Baselines/FFADE_YenChang_2021/queries
		cp ./anomalous_data/${dat}_${anom}_queries.txt ../../Baselines/MIDAS_SiddharthBhatia_2020/queries
		cp ./anomalous_data/${dat}_${anom}_queries.txt ../../Proposed/edge_anomaly/queries
		# labels
		cp ./anomalous_data/${dat}_${anom}_gt.txt ../../Baselines/FFADE_YenChang_2021/labels
		cp ./anomalous_data/${dat}_${anom}_gt.txt ../../Baselines/MIDAS_SiddharthBhatia_2020/labels
		cp ./anomalous_data/${dat}_${anom}_gt.txt ../../Proposed/edge_anomaly/labels
	done
done
