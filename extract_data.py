# Usage:
#=======================================================================#
#   python extract_data.py --input_dir inputs/ --output_dir outputs/
#=======================================================================#

import numpy as np
import os
import cv2
import glob
import shutil
import pytesseract
import re
import time
import argparse
from statistics import mode
import imutils
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local

queue = []


def apply_threshold(img, argument):
    switcher = {
        1: cv2.threshold(cv2.GaussianBlur(img, (9, 9), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        2: cv2.threshold(cv2.GaussianBlur(img, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        3: cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        4: cv2.threshold(cv2.medianBlur(img, 5), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        5: cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        6: cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
        7: cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
    }
    return switcher.get(argument, "Invalid method")


def crop_image(img, start_x, start_y, end_x, end_y):
    cropped = img[start_y:end_y, start_x:end_x]
    return cropped


def get_string(img_path, method):
    # Read image using opencv
    img = cv2.imread(img_path)
    file_name = os.path.basename(img_path).split('.')[0]
    file_name = file_name.split()[0]

    output_path = os.path.join(output_dir, file_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Crop the areas where liscence plate number is more likely present
    # img = crop_image(img, pnr_area[0], pnr_area[1], pnr_area[2], pnr_area[3])
    # img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)

    # CODE FROM scan.py
    ratio = img.shape[0] / 500.0

    orig = img.copy()
    img = imutils.resize(img, height=500)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # show the original image and the edge detected image
    #print("STEP 1: Edge Detection")
    #cv2.imshow("Image", img)
    #cv2.imshow("Edged", edged)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # loop over the contours
    for c in cnts:
            # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    
    # show the contour (outline) of the piece of paper
    #print("STEP 2: Find contours of paper")
    cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
    #cv2.imshow("Outline", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # apply the four point transform to obtain a top-down
    # view of the original image
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset=10, method="gaussian")
    warped = (warped > T).astype("uint8") * 255

    # show the original and scanned images
    #print("STEP 3: Apply perspective transform")
    #cv2.imshow("Original", imutils.resize(orig, height=250))
    #cv2.imshow("Scanned", imutils.resize(warped, height=250))
    #cv2.waitKey(0)

    img = warped.copy()

    # Convert to gray
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    #  Apply threshold to get image with only black and white
    img = apply_threshold(img, method)
    save_path = os.path.join(output_path, file_name +
                            "_filter_" + str(method) + ".jpg")
    cv2.imwrite(save_path, img)

    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(img, lang="eng")

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="This program extracts liscence numbers from images.")
    parser.add_argument("-i", "--input_dir",
                        help="Input directory for the files to be modified")
    parser.add_argument("-o", "--output_dir",
                        help="Output directory for the files to be modified")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    im_names = glob.glob(os.path.join(input_dir, '*.png')) + \
        glob.glob(os.path.join(input_dir, '*.jpg')) + \
        glob.glob(os.path.join(input_dir, '*.jpeg'))

    overall_start_t = time.time()
    for im_name in sorted(im_names):
        queue.append(im_name)

    print("The following images will be processed and the liscence plate numbers will be extracted: {}\n".format(queue))

    for im_name in im_names:
        start_time = time.time()
        print("*** The images that are in the queue *** \n{}\n".format(queue))

        print('#=======================================================')
        print(('# Image Currently in process {:s}'.format(im_name)))
        print('#=======================================================')
        queue.remove(im_name)
        file_name = im_name.split(".")[0].split("/")[-1]

        i = 1
        while i < 8:
            print("> The filter method " + str(i) + " is now being applied.")
            result = get_string(im_name, i)
            print("Extracted Text:")
            print(result)
            i += 1

        end_time = time.time()

        print('#=======================================================\n'
              '# Results finished for: ' + file_name + '\n'
              '#=======================================================\n'

              '# It took ' + str(end_time-start_time) + ' seconds.     \n'
              '#=======================================================\n')

    overall_end_t = time.time()

    print('#=======================================================\n'
          '# It took ' + str(overall_end_t-overall_start_t) + ' seconds.\n'
          '#=======================================================\n')
