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
//   Implements a fixed-size, hash-based count‐min sketch data structure used by
//   all MIDAS cores to maintain approximate counts of edge occurrences in 
//   constant time and space. Provides methods to hash keys, increment counters,
//   query minimum counts, and merge/reset sketches.
//
// Why keep it:
//   • Core dependency for every MIDAS variant (Normal, Filtering, Relational)  
//   • Encapsulates all CMS logic in one reusable, well‐tested class  
//   • Ensures O(r × d) memory bounds—critical for streaming performance  
//   • Decouples sketch implementation from anomaly-scoring logic  
//
// Key features:
//   • Parameterized rows (r) and columns (c) for tunable accuracy vs. memory  
//   • Randomized hash parameters for low collision bias  
//   • ClearAll/MultiplyAll to support decay or resets  
//   • Copy‐constructor for safe duplication of sketch state  
//
// Usage:
//   1. Construct with desired dimensions: `CountMinSketch sketch(r, c);`  
//   2. For each key, call `sketch.Hash(idxArr, a, b)` then  
//      `sketch.Add(idxArr)` or `query = sketch(idxArr);`  
//   3. Merge/reset between time ticks as needed in your core code.
// -----------------------------------------------------------------------------


#pragma once

#include <algorithm>

namespace MIDAS {
struct CountMinSketch {
	// Fields
	// --------------------------------------------------------------------------------

	const int r, c, m = 104729; // Yes, a magic number, I just pick a random prime
	const int lenData;
	int* const param1;
	int* const param2;
	float* const data;
	constexpr static float infinity = std::numeric_limits<float>::infinity();

	// Methods
	// --------------------------------------------------------------------------------

	CountMinSketch() = delete;
	CountMinSketch& operator=(const CountMinSketch& b) = delete;

	CountMinSketch(int numRow, int numColumn):
		r(numRow),
		c(numColumn),
		lenData(r * c),
		param1(new int[r]),
		param2(new int[r]),
		data(new float[lenData]) {
		for (int i = 0; i < r; i++) {
			param1[i] = rand() + 1; // ×0 is not a good idea, see Hash()
			param2[i] = rand();
		}
		std::fill(data, data + lenData, 0);
	}

	CountMinSketch(const CountMinSketch& b):
		r(b.r),
		c(b.c),
		lenData(b.lenData),
		param1(new int[r]),
		param2(new int[r]),
		data(new float[lenData]) {
		std::copy(b.param1, b.param1 + r, param1);
		std::copy(b.param2, b.param2 + r, param2);
		std::copy(b.data, b.data + lenData, data);
	}

	~CountMinSketch() {
		delete[] param1;
		delete[] param2;
		delete[] data;
	}

	void ClearAll(float with = 0) const {
		std::fill(data, data + lenData, with);
	}

	void MultiplyAll(float by) const {
		for (int i = 0, I = lenData; i < I; i++) // Vectorization
			data[i] *= by;
	}

	void Hash(int* indexOut, int a, int b = 0) const {
		for (int i = 0; i < r; i++) {
			indexOut[i] = ((a + m * b) * param1[i] + param2[i]) % c;
			indexOut[i] += i * c + (indexOut[i] < 0 ? c : 0);
		}
	}

	float operator()(const int* index) const {
		float least = infinity;
		for (int i = 0; i < r; i++)
			least = std::min(least, data[index[i]]);
		return least;
	}

	float Assign(const int* index, float with) const {
		for (int i = 0; i < r; i++)
			data[index[i]] = with;
		return with;
	}

	void Add(const int* index, float by = 1) const {
		for (int i = 0; i < r; i++)
			data[index[i]] += by;
	}
};
}
