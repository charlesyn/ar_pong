import numpy as np
import cv2

class HandSample:

    def __init__(self, rows, cols):
        self.x1 = np.array([9 * cols / 20, 9 * cols / 20, 11 * cols / 20, 11 * cols / 20]).astype(int)
        self.y1 = np.array([9 * rows / 20, 11 * rows / 20, 9 * rows / 20, 11 * rows / 20]).astype(int)
        self.x2 = self.x1 + 10
        self.y2 = self.y1 + 10
        self.num_rectangles = self.x1.size

    def draw_hand_sample(self, frame):
        for i in range(self.num_rectangles):
            cv2.rectangle(frame, (self.x1[i], self.y1[i]), (self.x2[i], self.y2[i]), (255, 0, 0), 2)
        return frame

    def get_hand_sample(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hand_pixels = np.zeros([40, 10, 3], dtype=hsv.dtype)

        for i in range(self.num_rectangles):
            hand_pixels[i * 10:i * 10 + 10, 0:10] = hsv[self.y1[i]:self.y2[i], self.x1[i]:self.x2[i]]

        hand_histogram = cv2.calcHist([hand_pixels], [0, 1], None, [180, 256], [0, 180, 0, 256])
        return hand_histogram
