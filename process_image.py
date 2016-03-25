'''
Created on Mar 23, 2016
Copied from: http://www.pyimagesearch.com/
2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/
other sources: # Take from: http://stackoverflow.com/a/28034418/2886003
@author: lluis
'''
from pyimagesearch import four_point_transform
import imutils
from skimage.filter import threshold_adaptive
import numpy as np
import argparse
import cv2


def resize_image(image, height=500):
    """Resize the image to the desired height"""
    image = imutils.resize(image, height=height)
    return image


def find_edges(image):
    """Find edges of the image in grayscale."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    return edged


def contours(edged):
    """Find the contours in the edged image."""
    (_, cnts, _) = cv2.findContours(edged.copy(),
                                 cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[: 5]

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
    else:
        raise ValueError("Make sure that the bill is in right shape!")
    return screenCnt


def extract_bill(image, screen, ratio):
    """"Extract the bill of the image"""
    warped = four_point_transform(image, screen.reshape(4, 2) * ratio)

    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped = threshold_adaptive(warped, 250, offset=10)
    warped = warped.astype("uint8") * 255
    return warped


def main(image):
    """Process the image"""
    image = cv2.imread(image)
    ratio = image.shape[0] / 500.0
    edges = find_edges(image)
    screen_c = contours(edges)
    warped = extract_bill(image, screen_c, ratio)
    return warped

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("image",
        help="Path to the image to be scanned")
    args = ap.parse_args()
    warped = main(args.image)

    # show the original and scanned images
#    print("STEP 3: Apply perspective transform")
#    cv2.imshow("Original", imutils.resize(image, height=650))
    cv2.imshow("Scanned", imutils.resize(warped, height=650))
    cv2.waitKey(0)
