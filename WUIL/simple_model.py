"""
light weight statistical anomaly detector which analyze user behavior over time, focusing on file path access patterns and timing difference 
Based on simple online learning of averages and variance of behavior and flags as malicious if any action is deviating too far 
Calculates the: 
    depth_delta: how much destination path differs from the source path 
    time_delta: time difference between the current event and last event 
Keeps a running average and variance of depth_delta and time_delta and for new event, calculates z-scores. This code does have threshold, and if z-score exceeds the threshold, it considers the event as malicious 
Learns only from nomral behavior 


"""

import math
from pathlib import Path
from sklearn.metrics import roc_auc_score

class SimpleModel:

    # Notes for change:
    # some damping factor for the avg? Similar to that of MIDAS
    # benign users often revisit the same folders (you ever only use some specific folders as a normal user)
    #   extra feature: weight of the edge - if weight is low then anomaly score should be high
    #   either sliding window on edges or damping factor
    # burst of edges?
    # use CMS to keep track of weight
    
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

        # To avoid division by zero errors
        if self._depth_var == 0:
            self._depth_var = 0.001
        if self._time_var == 0:
            self._time_var = 0.001

        depth_z_score = ((depth_delta - self._depth_avg))/math.sqrt(self._depth_var)
        time_z_score = ((time_delta - self._time_avg))/math.sqrt(self._time_var)

        # Find the sum of deviations from mean (z-score)
        sum_z = (abs(depth_z_score) + abs(time_z_score))
        sum_z = abs(depth_z_score)

        # Predict
        if sum_z/2 > self._threshold:
            return True
        else:
            return False

    """
    Function to update the average and variance of depth delta and time delta
    """
    def update_model(self, depth_delta, time_delta, size):

        # Calculate the new averages
        new_depth_avg = self._depth_avg + (depth_delta-self._depth_avg)/(size+1)
        new_time_avg = self._time_avg + (time_delta-self._time_avg)/(size+1)
        
        # Calculate the new variances
        new_depth_var = ((size-1) * self._depth_var + (depth_delta-self._depth_var) * (depth_delta-new_depth_avg))/(size)
        new_time_var = ((size-1) * self._time_var + (time_delta-self._time_var) * (time_delta-new_time_avg))/(size)

        # Update the metrics
        self._depth_avg = new_depth_avg
        self._time_avg = new_time_avg

        self._depth_var = new_depth_var
        self._time_var = new_time_var

    """
    Function to calculate the longest common path parent directory
    """
    def commonpath(self, path1: Path, path2: Path) -> Path:
        i = 0
        common_path = Path("")

        # Iterate until the parent directory does not match
        while i < len(path1.parts) and i < len(path2.parts):
            if path1.parts[i] == path2.parts[i]:
                common_path = common_path / path1.parts[i]
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

            # Iterate through every line in the data file
            for line in f:
                size += 1
                data = line.split(",")
                if len(data) != 4:
                    print("Invalid data of simple_model. Must be in the form [timestamp, src_node, dst_node, label]")
                    return
                
                # Unpack the data
                timestamp = int(data[0])
                src_node = Path(data[1])
                dst_node = Path(data[2])
                
                label = data[3]
                labels.append(int(label))
                
                # Make sure prev_time is instantiated
                if size == 1:
                    prev_time = timestamp
                
                # Ignore data when user is idle
                if dst_node == src_node:
                    predictions.append(0)
                    size -= 1
                else:

                    # Find time difference and path distance
                    depth_delta = len(Path(dst_node).parts) + len(Path(src_node).parts) - 2*len(self.commonpath(src_node, dst_node).parts)
                    time_delta = int(timestamp) - prev_time
                    
                    # Instantiate the average and variance metrics with the first two data points (idle data is ignored)
                    if size < 3:
                        self.update_model(depth_delta, time_delta, size)
                        predictions.append(0)
                    else:
                        is_mal = self.classify(depth_delta, time_delta)
                        predictions.append(int(is_mal))
                        
                        # Update the model only if the data is not malicious
                        if not is_mal:
                            self.update_model(depth_delta, time_delta, size)
        
                prev_time = timestamp
        print(roc_auc_score(labels, predictions), self._depth_avg, self._time_avg)




# Sanity check
model = SimpleModel(2)
model.run_model("./custom_data/simple_model_data.csv")
model.run_model("./custom_data/simple_model_benign.csv")


                    






            
        
