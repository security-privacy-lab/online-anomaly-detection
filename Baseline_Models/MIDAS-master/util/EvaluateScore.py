# ------------------------------------------------------------------------------
# Copyright 2020 Rui Liu (liurui39660) and Siddharth Bhatia (bhatiasiddharth)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------
# Purpose:
#   Computes the ROC-AUC between the ground-truth labels and anomaly scores,
#   prints the result, and optionally writes it to a file for automated
#   experiment pipelines.
#
# Why keep it:
#   • Standardizes evaluation: ensures all MIDAS variants are scored
#     consistently using the same metrics and thresholds  
#   • Integrates seamlessly with the demo and experiment drivers
#     (`Demo.cpp`, `Experiment.cpp`) via system calls  
#   • Simplifies reproducibility: users need only run this script to
#     get AUC values in both human-readable and machine-parsable form  
#
# Usage:
#   python util/EvaluateScore.py <pathGroundTruth> <pathScore> [<indexRun>]
#
#   - <pathGroundTruth>: CSV file of true labels (0/1)  
#   - <pathScore>:      CSV file of anomaly scores (one float per line)  
#   - [<indexRun>]:     Optional run index; if provided, AUC is also
#                       saved to `temp/AUC<indexRun>.txt`
#
# Output:
#   Prints “ROC-AUC<indexRun> = <value>” to stdout and writes the
#   AUC to `temp/AUC<indexRun>.txt` when `indexRun` is given.
# ------------------------------------------------------------------------------
from pathlib import Path
from sys import argv

from pandas import read_csv
from sklearn.metrics import roc_auc_score

root = (Path(__file__) / '../..').resolve()

if len(argv) < 3:
	print('Print ROC-AUC to stdout and MIDAS/temp/AUC[<indexRun>].txt')
	print('Usage: python EvaluateScore.py <pathGroundTruth> <pathScore> [<indexRun>]')
else:
	y = read_csv(argv[1], header=None)
	z = read_csv(argv[2], header=None)
	indexRun = argv[3] if len(argv) >= 4 else ''
	auc = roc_auc_score(y, z)
	print(f"ROC-AUC{indexRun} = {auc:.4f}")
	if indexRun:
		with open(root / f"temp/AUC{indexRun}.txt", 'w') as file:
			file.write(str(auc))
