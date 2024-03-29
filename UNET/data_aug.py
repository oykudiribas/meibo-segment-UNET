import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
#import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate


""" Create a directory """
# def create_dir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)


def load_data():
    # train_x = sorted(glob(os.path.join(path, "training", "image-train", "*.JPG")))[:20]
    # train_y = sorted(glob(os.path.join(path, "training", "mask-train", "*.png")))[:20]


    # test_x = sorted(glob(os.path.join(path, "test", "image-test", "*.JPG")))
    # test_y = sorted(glob(os.path.join(path,"test", "mask-test", "*.png")))

    image_folder="data_all/images"
    mask_folder="data_all/eyelid"
    gland_folder="data_all/glands"

    image_files = sorted(os.listdir(image_folder))
    mask_files = sorted(os.listdir(mask_folder))
    gland_files = sorted(os.listdir(gland_folder))

    # # Split into train and test
    train_x = [os.path.join(image_folder, f) for f in image_files]
    # train_y = [os.path.join(mask_folder, f) for f in mask_files]
    train_y = [os.path.join(gland_folder, f) for f in gland_files]

    test_x = [os.path.join(image_folder, f) for f in image_files]
    # test_y = [os.path.join(mask_folder, f) for f in mask_files]
    test_y = [os.path.join(gland_folder, f) for f in gland_files]


    

    return (train_x, train_y), (test_x, test_y)


   

def augment_data(images, masks, save_path, augment=True):
    size = (256,128)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting the name """
        print(flush=True)
        filename_x = os.path.basename(x)
        # print(filename_x)
        filename_y = os.path.basename(y)
        # print(filename_y)

        name= os.path.splitext(filename_x)[0]
        # file_y = os.path.splitext(filename_y)[0]
        # print(file_x)
        # print(file_y)
        # print(name)
        """ Reading image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y_img1 = cv2.imread(y)
        y=cv2.cvtColor(y_img1,cv2.COLOR_RGB2GRAY)
        # y_img = imageio.mimread(y)[0]
        # print(x.shape, y.shape)
     
        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = Rotate(limit=5, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]

        

            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in (zip(X, Y)):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            tmp_image_name = f"{name}_{index}.JPG"
            tmp_gland_name = f"{name}_{index}.png"
            # tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            gland_path = os.path.join(save_path, "gland", tmp_gland_name)
            # mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(gland_path, m)
            # cv2.imwrite(mask_path, m)

            index += 1

      

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    #data_path = "data_for_meibo"
    #data_path = "data_all"
    (train_x, train_y), (test_x, test_y) = load_data()

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Create directories to save the augmented data """
    # create_dir("new_data/train/image/")
    # create_dir("new_data/train/mask/")
    # create_dir("new_data/test/image/")
    # create_dir("new_data/test/mask/")

    """ Data augmentation """
    augment_data(train_x, train_y, "new_data/train/", augment=True)
    augment_data(test_x, test_y, "new_data/test/", augment=False)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

