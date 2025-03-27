import pandas as pd
from torch.utils.data import Dataset
import numpy as np

pd.options.mode.chained_assignment = None

MAX_DISPLACEMENT = 40

def read_dataset(datasets : str, type : str):
        
        
        df = pd.read_csv(datasets)

        x = df[["x", "y","dx", "dy"]] 
        targets = df[["x_to", "y_to"]]
        y = construct_ground_truth(df[["x", "y"]], df[["x_to", "y_to"]], type)

        x.fillna(0, inplace=True)

        return x, y, targets

class FittsDataset(Dataset):
    def __init__(self,x, y):        
        self.data = x
        self.y_gt = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return self.data.iloc[idx, :].to_numpy(), self.y_gt.iloc[idx, :].to_numpy()
        return self.data[idx, :], self.y_gt[idx, :]
    
class FittsDatasetSeq(Dataset):
    def __init__(self,x, y, sequence_length):        
        self.data = x
        self.y_gt = y
        self.seq_l = sequence_length

    def __len__(self):
        return len(self.data) - self.seq_l

    def __getitem__(self, idx):
        # return self.data.iloc[idx:idx+self.seq_l, :].to_numpy(), self.y_gt.iloc[idx + self.seq_l, :].to_numpy()
        return self.data[idx:idx+self.seq_l, :], self.y_gt[idx + self.seq_l, :]

def construct_ground_truth(cursor_pose, target_pose, type):
    """
        
        - vec ground truth is the best dx between Ct and Tt it is the believed  
            (IE at each instant it s a straight line leading to the target)
        - second type is an smoothed version of the trajectory to fit a more natural lines between positions and targets
    """
    y = {}
    if type == "vec":
        # doesn't work, clipping the displacement change direction :(
        dx = (target_pose['x_to'] - cursor_pose['x'])
        dy = (target_pose['y_to'] - cursor_pose['y'])
        mag = np.clip(np.sqrt(dx**2 + dy**2), 0, MAX_DISPLACEMENT)
        angle = np.arctan2(dy, dx)

        y['dx'] = mag * np.cos(angle)
        y['dy'] = mag * np.sin(angle)


        # target_pose['x_to'] - cursor_pose['x']
    # elif type == "smooth":
    #     y['dx'] = target_pose['x_to'] - cursor_pose['x']
    #     y['dy'] = target_pose['y_to'] - cursor_pose['y']
    return pd.DataFrame(y)

    

if __name__ == "__main__":
    x, y =read_dataset("/home/jmartinsaquet/Documents/code/IA2_codes/clone/datasets/P0_C1.csv", "vec")

    print("x : \n", x)
    print("y : \n", y)

    print("y mag : \n", np.sqrt(y['dx']**2 + y['dy']**2))

