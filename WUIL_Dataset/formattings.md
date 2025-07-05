# Attack Injection File Naming Format

This repository uses a structured naming format to describe injected attacks for anomaly detection evaluation. Each filename provides key information about the algorithm used, the attack scenario, the injection method, and the injection position.

---

## File Naming Convention
<algorithm_type> <attack_number> <injection_method> <injection_position>.txt


algorithm_type:
- The detection algorithm used
- anomrank for Anomrank
- midas for MIDAS

attack_number: 
- Attack log number used(eg., attack1, attack2, or attack3)

injection_method: 
- **5m** for 5-minute continuous segment injection
- **org** for organized injection

position:
- approximate location of the injected segment within the original log in percentage of 0 - 100.
- For instance, 61 means the attack was injected at 61% into the file.

---
### Example:
  anomrank_attack1_5m_61.txt
  

## Components Explained

| Component             | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| algorithm_type      | The anomaly detection algorithm used. <br>• anomrank for AnomRank <br>• midas for MIDAS |
| attack#             | Attack log number used (e.g., attack1, attack2, or attack3)           |
| injection_method    | How the attack was injected: <br>• 5m = 5-minute continuous segment <br>• org = organized/injected in segments across the log |
| injection_position  | Approximate location in the user log (in percentage of total file length). For example, 61 means the attack was injected starting at 61% of the log. |

---

## Example Interpretations

- anomrank_attack1_5m_61.txt  
  → AnomRank algorithm, Attack 1 log injected as a 5-minute block at 61% through the original log.



User 16: Longest benign behavior baseline 

User 8: The longest attack1 file 

User 7: The longest attack 2 file + longest attacker 1,2,3 combined. 

User 1: longest attack 3 file 

User 12: user that had most heavily sanitized log 


-----

## File Processing(Preprocess) 
Anomrank File has been created by the file "data_creation_by_one_run.py" inside online-anomaly-detection\WUIL\ directory. The original WUIL formatting does not suit for the MIDAS or AnomRank therefore the hashing was required

**Midas**: The file requires 3 separate files for one run. 
- Meta file(shape_file): the integer N, the number of the records in the dataset
- Data file: The header-less csv format of the shape; columns being src, dst, timestamp
- Label File: The label for the data, 0 meaning normal and 1 meaning anomalous.
As Midas accepts the integer for the src and dst, had to hash them, and use session ID for the timestamp

| Steps             | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| Step 1: Load the Dataset      | Loads the data set that is in ID|Date|Time|Session_ID|Depth|Path|Label and sort them by Session_ID and force label 0 on benign dataset and 1 on the malicious(attacker) datasets |
| Step 2: Perform Injection| Inject attacks by merging the malicious rows into the benign stream using either 5-minute block(where session_ID jumps >= 301) or ordered(simply append & resort by Session_ID           |
| Step 3:  Edge-list construction  | It sort by the Session_ID -> Path log as AnomRank |
| Step 4: Dense ID remapping| Concatnates edges of src_node and dst_node and extract the unique values and build a python dictionary and replace the both src_node and dst_node by factor_mapping therefore node ID becomes readable by MIDAS algorithm |
| Step 5: Write Midas File | As Preprocessing is done, we write the 3 files, in feature(src, dst, time), label(0/1) and shape(number of edge)|


**AnomRank**: The file requires 1 file for one run 
- Each columns means: timestamp src dst label

| Steps             | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| Step 1: Load the Dataset      | Loads the data set that is in ID|Date|Time|Session_ID|Depth|Path|Label and sort them by Session_ID and force label 0 on benign dataset and 1 on the malicious(attacker) datasets |
| Step 2: Perform Injection| Inject attacks by merging the malicious rows into the benign stream using either 5-minute block(where session_ID jumps >= 301) or ordered(simply append & resort by Session_ID           |
| Step 3:  Make directed-edge list  | Walk through the merged log in order, keeping track of previous Path. prev_Path -> current_path, timestamp, label. If  current_Path == prev_path, it does self-loop |
| Step 4: Normalize & remap | find the minimum Session_ID and subtract it from every session_ID therefore first event is at 1(to avoid the error by integer being too big) and Map the unique string path to a dense integer node ID as they first appear |
| Step 5: Write AnomRank File | As Preprocessing is done, we write the file in output format <timestamp><src_node><dst_node><label>|
