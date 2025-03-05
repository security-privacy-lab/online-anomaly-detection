import math
from pathlib import Path
from sklearn.metrics import roc_auc_score

class SimpleModel:

    def __init__(self, threshold):
        self._threshold = threshold
        
        self._depth_avg = 0
        self._depth_var = 0
        
        self._time_avg = 0
        self._time_var = 0

    """
    Function to determine if data is malicious (True) or not (False)
    """
    def classify(self, depth_delta, time_delta):
        depth_z_score = ((depth_delta - self._depth_avg)**2)/math.sqrt(self._depth_var)
        time_z_score = ((time_delta - self._time_avg)**2)/math.sqrt(self._time_var)

        sum_z = (abs(depth_z_score) + abs(time_z_score))

        if sum_z > self._threshold:
            return True
        else:
            return False

    """
    Function to update the average and variance of depth delta and time delta
    """
    def update_model(self, depth_delta, time_delta, size):
        new_depth_avg = self._depth_avg + (depth_delta-self._depth_avg)/(size+1)
        new_time_avg = self._time_avg + (time_delta-self._time_avg)/(size+1)
        
        new_depth_var = ((size-1) * self._depth_var + (depth_delta-self._depth_var) * (depth_delta-new_depth_avg))/(size)
        new_time_var = ((size-1) * self._time_var + (time_delta-self._time_var) * (time_delta-new_time_avg))/(size)


        self._depth_avg = new_depth_avg
        self._time_avg = new_time_avg

        self._depth_var = new_depth_var
        self._time_var = new_time_var

    def commonpath(self, path1: Path, path2: Path) -> Path:
        i = 0
        common_path = Path("")
        while i < len(path1.parts) and i < len(path2.parts):
            if path1.parts[i] == path2.parts[i]:
                common_path = common_path.joinpath(path1.parts[i])
            else:
                break
            i += 1
        return common_path


    """
    Function to consume each time series data in the form [timestamp, src_node, dst_node, label] of the file specifed by filename
    """
    def run_model(self, filename):
        predictions = []
        labels = []
        with open(filename, 'r') as f:
            size = 0
            prev_time = 0
            for line in f:
                size += 1
                data = line.split(",")
                if len(data) != 4:
                    print("Invalid data fof simple_model. Must be in the form [timestamp, src_node, dst_node, label]")
                    return
                # Unpack the group data
                timestamp = data[0]
                src_node = Path(data[1])
                dst_node = Path(data[2])
                
                label = data[3]
                labels.append(int(label))
                
                if size == 1:
                    prev_time = int(timestamp)
                
                depth_delta = len(Path(dst_node).parts) - len(self.commonpath(src_node, dst_node).parts)
                time_delta = int(timestamp) - prev_time
                if size < 100:
                    self.update_model(depth_delta, time_delta, size)
                    predictions.append(0)
                else:
                    is_mal = self.classify(depth_delta, time_delta)
                    predictions.append(int(is_mal))
                    
                    if not is_mal:
                        self.update_model(depth_delta, time_delta, size)
        
        print(roc_auc_score(labels, predictions))




# Sanity check
model = SimpleModel(2)
model.run_model("./custom_data/simple_mode_data.csv")


                    






            
        