import cv2 as cv
import numpy as np
import math

img = np.zeros((400, 700, 1), dtype = "uint8")
print(img.shape)

cv.imshow("Input", img)
cv.waitKey(0)

lrMin = 250
lrMax = 100
iterations = 400
stepsize = 100
cycle = math.floor(1 + (iterations/(2*stepsize)))
lrstart = 50
lrend = lrstart+stepsize

for i in range(cycle):
    # x = abs((iterations/stepsize) - (2*cycle) + 1)
    # lrt = lrMin + ((lrMax - lrMin)*(1-x))
    # print(x, lrt)
    cv.line(img, (lrstart, lrMin), (lrend, lrMax), (255,0,0), 5)
    lrstart = lrend
    lrend = lrstart+stepsize
    cv.line(img, (lrstart, lrMax), (lrend,lrMin), (255,0,0), 5)
    lrstart = lrend
    lrend = lrstart+stepsize


cv.imshow("Curve", img)
cv.waitKey(0)

cv.imwrite("curve.png", img)


