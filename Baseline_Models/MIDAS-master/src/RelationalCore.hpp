// -----------------------------------------------------------------------------
// Copyright 2020 Rui Liu (liurui39660) and Siddharth Bhatia (bhatiasiddharth)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// 	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// -----------------------------------------------------------------------------
// Purpose:
//   Implements the MIDAS-R (relational) variant by extending the core sketching
//   algorithm to capture bursts not only on individual edges but also on all
//   edges incident to a source or destination node. Maintains three pairs of
//   Count–Min Sketches (edge, source, destination) for current and total counts.
//
// Why keep it:
//   • Provides enhanced detection of coordinated or relational anomalies that
//     span multiple edges around a node  
//   • Serves as the reference implementation for the AAAI 2020 “MIDAS-R” method  
//   • Demonstrates how to integrate multi‐sketch scoring in a single core  
//   • Allows easy comparison against NormalCore and FilteringCore in experiments  
//
// Key features:
//   • Separate sketch pairs for edges, sources, and destinations  
//   • Exponential decay of current‐tick counts via a user‐specified factor  
//   • Computes final score as the maximum deviation across all three sketches  
//   • O(r × d) memory and O(r) time per edge, independent of graph size  
//
// Usage:
//   1. Construct with desired sketch dimensions and decay factor, e.g.  
//        `RelationalCore core(numRows, numCols, decayFactor);`  
//   2. For each incoming edge (u, v, t), call:  
//        `float score = core(u, v, t);`  
//   3. Scores above your chosen threshold indicate relational bursts/anomalies.
// -----------------------------------------------------------------------------
#pragma once

#include <cmath>

#include "CountMinSketch.hpp"

namespace MIDAS {
struct RelationalCore {
	int timestamp = 1;
	const float factor;
	int* const indexEdge; // Pre-compute the index to-be-modified, thanks to the same structure of CMSs
	int* const indexSource;
	int* const indexDestination;
	CountMinSketch numCurrentEdge, numTotalEdge;
	CountMinSketch numCurrentSource, numTotalSource;
	CountMinSketch numCurrentDestination, numTotalDestination;

	RelationalCore(int numRow, int numColumn, float factor = 0.5):
		factor(factor),
		indexEdge(new int[numRow]),
		indexSource(new int[numRow]),
		indexDestination(new int[numRow]),
		numCurrentEdge(numRow, numColumn),
		numTotalEdge(numCurrentEdge),
		numCurrentSource(numRow, numColumn),
		numTotalSource(numCurrentSource),
		numCurrentDestination(numRow, numColumn),
		numTotalDestination(numCurrentDestination) { }

	virtual ~RelationalCore() {
		delete[] indexEdge;
		delete[] indexSource;
		delete[] indexDestination;
	}

	static float ComputeScore(float a, float s, float t) {
		return s == 0 || t - 1 == 0 ? 0 : pow((a - s / t) * t, 2) / (s * (t - 1));
	}

	float operator()(int source, int destination, int timestamp) {
		if (this->timestamp < timestamp) {
			numCurrentEdge.MultiplyAll(factor);
			numCurrentSource.MultiplyAll(factor);
			numCurrentDestination.MultiplyAll(factor);
			this->timestamp = timestamp;
		}
		numCurrentEdge.Hash(indexEdge, source, destination);
		numCurrentEdge.Add(indexEdge);
		numTotalEdge.Add(indexEdge);
		numCurrentSource.Hash(indexSource, source);
		numCurrentSource.Add(indexSource);
		numTotalSource.Add(indexSource);
		numCurrentDestination.Hash(indexDestination, destination);
		numCurrentDestination.Add(indexDestination);
		numTotalDestination.Add(indexDestination);
		return std::max({
			ComputeScore(numCurrentEdge(indexEdge), numTotalEdge(indexEdge), timestamp),
			ComputeScore(numCurrentSource(indexSource), numTotalSource(indexSource), timestamp),
			ComputeScore(numCurrentDestination(indexDestination), numTotalDestination(indexDestination), timestamp),
		});
	}
};
}
