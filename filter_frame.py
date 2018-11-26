import numpy as np
import cv2
import math

class FilterFrame:

    def apply_filter(self, frame, histogram):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0, 1], histogram, [0, 180, 0, 256], 1)

        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
        dst = cv2.filter2D(dst, -1, disc, dst)
        erode_kernel = np.ones((3,3),np.uint8)
        dilate_kernel = np.ones((5,5), np.uint8)
        dst = cv2.erode(dst, erode_kernel, iterations=1)

        return dst

    def getFingertips(self, frame, contour):
        hull = cv2.convexHull(contour, returnPoints = False)

        defects = cv2.convexityDefects(contour, hull)

        if defects is None:
            return frame

        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            sPrev, ePrev, fPrev, dPrev = defects[i - 1, 0]
            sNext, eNext, fNext, dNext = defects[(i + 1) % defects.shape[0], 0]
            start = tuple(contour[s][0])
            nextStart = tuple(contour[sNext][0])
            prevStart = tuple(contour[sPrev][0])
            if self.findAngle(start, nextStart, prevStart) > 0.6:
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                cv2.line(frame,start,end,[0,255,0],2)
                frame = cv2.circle(frame,far,5,[0,0,255],-1)

        return frame

    def findAngle(self, a, b, c):
        vecAB = self.findDistance(a, b)
        vecBC = self.findDistance(b, c)
        vecAC = self.findDistance(a, c)
        print(math.acos((vecAB**2 + vecBC**2 - vecAC**2) / (2 * vecAB * vecBC)))
        return math.acos((vecAB**2 + vecBC**2 - vecAC**2) / (2 * vecAB * vecBC))

    def findDistance(self, a, b):
        sum = 0
        for i in range(len(a)):
            sum = sum + (a[i] - b[i])**2
        return sum**(1/2.0)
