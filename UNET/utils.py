
import os
import time
import random
import numpy as np
import cv2
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    if device == 'cpu':
        torch.manual_seed(seed)
    
    #For GPU

    else:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

""" Create a directory. """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Calculate the time taken """
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs