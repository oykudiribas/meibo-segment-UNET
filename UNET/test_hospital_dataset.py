import os, time
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
from histmatch import clear_output_directory
from model import build_unet
from utils import create_dir, seeding

clear_output_directory("results")

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Folders """
    # create_dir("results")

    """ Load dataset """

    test_folder="new_raw"
    # test_folder="new_data/test/image/"
  

    test_files = sorted(os.listdir(test_folder))
    

    # # Split into train and test
    test_x = [os.path.join(test_folder, f) for f in test_files]
    x=test_x[2]
    y=cv2.imread(x)
    print(y.shape)

    # test_x = sorted(glob("new_data/test/image/*"))[900:910]

   
    # test_y = sorted(glob("new_data/test/gland/*"))[900:1000]

    # """ Hyperparameters """
    W = 1280
    H = 640
    
    size = (W, H)
    checkpoint_path = "files/checkpoint.pth"

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    for i, x in tqdm(enumerate(test_x), total=len(test_x)):
        """ Extract the name """
        print(flush=True)
        filename_x = os.path.basename(x)
        name= os.path.splitext(filename_x)[0]
        
        # Reading image
        image = cv2.imread(x, cv2.IMREAD_COLOR) ## (512, 512, 3)
        image = cv2.resize(image, size)
        x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
        x = x/255.0
        x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)

        with torch.no_grad():
            """ Prediction """
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
            pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        """ Saving masks """
        pred_y = mask_parse(pred_y)

        

        
        line = np.ones((size[1], 10, 3)) * 128

        cat_images = np.concatenate(
            [image,line, pred_y * 255], axis=1
        )
        cv2.imwrite(f"results/{name}.png", cat_images)
        