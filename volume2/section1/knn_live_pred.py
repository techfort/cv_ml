import cv2
from knn_train_mnist import get_trained_knn
import numpy as np

def inside(r1, r2):
  x1,y1,w1,h1 = r1
  x2,y2,w2,h2 = r2
  if (x1 > x2) and (y1 > y2) and (x1+w1 < x2+w2) and (y1+h1 < y2 + h2):
    return True
  else:
    return False

def wrap_digit(rect):
  x, y, w, h = rect
  padding = 5
  hcenter = x + w/2
  vcenter = y + h/2
  roi = None
  if (h > w):
    w = h
    x = hcenter - (w/2)
  else:
    h = w
    y = vcenter - (h/2)
  return (x-padding, y-padding, w+padding, h+padding)

KNN = get_trained_knn()
kernel = np.ones((3,3),np.uint8)
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
area = 0
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    frame = cv2.imread("./digits.png", cv2.IMREAD_COLOR)

    if frame is not None:
        area = frame.shape[0] * frame.shape[1]
        thbw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        """
    	thbw = cv2.GaussianBlur(bw, (3,3), 0)
        thbw = cv2.threshold(thbw, 127, 255, cv2.THRESH_BINARY)[1]
        thbw = cv2.dilate(thbw, es, iterations=2)
        """
        thbw = cv2.erode(thbw, kernel, iterations = 2)
        cv2.imshow("threshold", thbw)
        image, cntrs, hier = cv2.findContours(thbw.copy(), cv2.RETR_EXTERNAL
                , cv2.CHAIN_APPROX_SIMPLE)
        rectangles = []

	for c in cntrs:
  	    r = x,y,w,h = cv2.boundingRect(c)
  	    a = cv2.contourArea(c)
  	    b = (image.shape[0]-3) * (image.shape[1] - 3)
  
  	    is_inside = False
  	    for q in rectangles:
    		if inside(r, q):
      		    is_inside = True
      		    break
  	    if not is_inside:
    	        if not a == b:
      	            rectangles.append(r)

        for r in rectangles:
	    x,y,w,h = wrap_digit(r) 
  	    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
  	    roi = thbw[y:y+h, x:x+w]
  	    sample = cv2.flip(np.array([np.resize(roi,(28,28)).ravel()], dtype=np.float32),1)
    	    print sample.shape
    	    digit_class = KNN.findNearest(sample, k=3)[1][0][0]
  	    print digit_class
  	    cv2.putText(frame, "%d" % digit_class, (x, y + h/2), font, 1, (0, 255, 0))
        
        cv2.imshow("camera", frame)
        if cv2.waitKey() == ord("q"):
            break

cv2.destroyAllWindows()
