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
| `algorithm_type`      | The anomaly detection algorithm used. <br>• `anomrank` for AnomRank <br>• `midas` for MIDAS |
| `attack#`             | Attack log number used (e.g., `attack1`, `attack2`, or `attack3`)           |
| `injection_method`    | How the attack was injected: <br>• `5m` = 5-minute continuous segment <br>• `org` = organized/injected in segments across the log |
| `injection_position`  | Approximate location in the user log (in percentage of total file length). For example, `61` means the attack was injected starting at 61% of the log. |

---

## Example Interpretations

- `anomrank_attack1_5m_61.txt`  
  → AnomRank algorithm, Attack 1 log injected as a 5-minute block at 61% through the original log.



User 16: Longest benign behavior baseline 

User 8: The longest attack1 file 

User 7: The longest attack 2 file + longest attacker 1,2,3 combined. 

User 1: longest attack 3 file 

User 12: user that had most heavily sanitized log 

