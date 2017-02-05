import cv2
import numpy as np

detector = cv2.xfeatures2d.SIFT_create(25)
extractor = cv2.xfeatures2d.SIFT_create()

def extract_features(img):
    return np.array(np.ravel(extractor.compute(img, detector.detect(img))[1]),
            dtype=np.float32)
