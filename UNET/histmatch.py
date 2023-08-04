import os
import cv2
import numpy as np

def clear_output_directory(output_directory):
    file_list = os.listdir(output_directory)
    for filename in file_list:
        file_path = os.path.join(output_directory, filename)
        os.remove(file_path)

def resize_image(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def histogram_match(input_image, reference_image):
    # Load input and reference images
    input_img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    reference_img = cv2.imread(reference_image, cv2.IMREAD_GRAYSCALE)

    # Calculate histograms of input and reference images
    hist_input = cv2.calcHist([input_img], [0], None, [256], [0, 256])
    hist_reference = cv2.calcHist([reference_img], [0], None, [256], [0, 256])

    # Compute the cumulative distribution functions (CDFs)
    cdf_input = hist_input.cumsum() / hist_input.sum()
    cdf_reference = hist_reference.cumsum() / hist_reference.sum()

    # Perform histogram matching by mapping the intensity values
    mapping = np.interp(cdf_input, cdf_reference, np.arange(256))
    matched_image = mapping[input_img]

    return matched_image

# Dosya yollarını ayarlayın
input_directory = 'raw'
output_directory = 'new_raw'
reference_dir = "new_data/test/image"
ref_list = os.listdir(reference_dir)

clear_output_directory(output_directory)

# # El ile belirttiğiniz index değeri
# selected_index = 1

# if selected_index < len(ref_list):
#     reference_image = os.path.join(reference_dir, ref_list[selected_index])
#     print("Reference image path:", reference_image)
# else:
#     print("Selected index is out of range.")

reference_image="ref.JPG"

# Girdi dizinindeki dosya listesini alın
file_list = os.listdir(input_directory)

# Histogram eşleştirme işlemini yapın ve fotoğrafları kaydedin
for filename in file_list:
    input_path = os.path.join(input_directory, filename)
    output_path = os.path.join(output_directory, filename)

    # Histogram eşleştirme işlemini yapın
    input_image=cv2.imread(input_path,cv2.IMREAD_GRAYSCALE)
    # matched_image=cv2.equalizeHist(input_image)
    matched_image = histogram_match(input_path, reference_image).astype(np.uint8)

    # resized_image = resize_image(matched_image, width=256, height=128)

    # equalized_image=cv2.equalizeHist(resized_image)

    # Yeniden boyutlandırılmış fotoğrafı kaydedin
    cv2.imwrite(output_path, matched_image)

  

    print(f"{filename} için histogram eşleştirme tamamlandı ve {output_path} adresine kaydedildi.")

