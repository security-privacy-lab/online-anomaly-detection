# Attack Injection & Preprocessing Guide

This document explains:

1. **Attack injection file naming conventions**
2. **Preprocessing pipeline** for AnomRank and MIDAS

---

## 1. File Naming Convention

Filenames encode algorithm, attack scenario, injection method, and injection position:

```
<algorithm>_<attack#>_<method>_<position>.txt
```

* `algorithm`: `anomrank` or `midas`
* `attack#`: `attack1`, `attack2`, `attack3`, or `combined`
* `method`:

  * `5m` — a single continuous 5‑minute block injected
  * `org` — attacks appended and re‑sorted by time
* `position`: integer 0–100 indicating the approximate percentile into the benign log where injection starts

**Example:**

```
anomrank_attack1_5m_61.txt
```

→ AnomRank, using Attack1 logs in a 5‑minute block at 61% through the baseline timeline.

---

## 2. Preprocessing Pipeline

We begin with LANL logs in the WUIL format:

```
ID|Date|Time|Session_ID|Depth|Path|Label
```

and produce algorithm‑ready edge lists.

### 2.1 Common Steps

1. **Load & Label**

   * Read `*.txt` with `|` delimiter into pandas.
   * Sort by `Session_ID`.
   * Force `Label=0` for benign dataset, `Label=1` for attacker dataset.
2. **Inject Attacks**

   * **5m:** find gaps (Session\_ID jumps ≥301), inject full attack log at chosen gap.
   * **org:** append entire attack log(s), resort by `Session_ID`.

### 2.2 MIDAS Preprocessing

MIDAS requires three files per run:

* **Shape file** (`*_shape.txt`): number of edges (N).
* **Features file** (`*_features.csv`): headerless CSV of `src,dst,timestamp`.
* **Labels file** (`*_labels.csv`): headerless CSV of the binary labels.

**Steps:**

1. **Edge‑List Construction:** emit `(prev_path → current_path, Session_ID, Label)` per row.
2. **ID Remapping:** gather all unique `src`/`dst` values, map them to integers `0…M-1`.
3. **Write Files:**

   * `features.csv`, `labels.csv`, and `shape.txt` as described.

### 2.3 AnomRank Preprocessing

AnomRank requires a single edge list file:

```
<timestamp> <src_node> <dst_node> <label>
```

**Steps:**

1. **Edge‑List Construction:** same as above.
2. **Normalize Timestamps:** subtract min(`Session_ID`) and add 1 so events start at `1`.
3. **Node Mapping:** assign each unique `Path` string an integer ID on first appearance.
4. **Write File:** export space‑delimited `<timestamp> <src> <dst> <label>.txt`.

---

With these edge lists in place, you can run:

```bash
./anomrank <file>.txt " " <window> <warmup> 0 0 0
```

or for MIDAS, load the three generated files per its API.
