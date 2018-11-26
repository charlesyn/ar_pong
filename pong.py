import numpy as np
import cv2
import hand_sample
import filter_frame

cap = cv2.VideoCapture(0)
rows = 480
cols = 640
hs = hand_sample.HandSample(rows, cols)

while True:
    ret, frame = cap.read()
    original = frame.copy()
    hs.draw_hand_sample(frame)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        histogram = hs.get_hand_sample(original)
        break

ff = filter_frame.FilterFrame()

while True:
    ret, frame = cap.read()
    dst = ff.apply_filter(frame, histogram)
    im2, contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sortedContours = np.array(sorted(contours, key=cv2.contourArea))
    biggestContour = sortedContours[-1:]
    cnt = sortedContours[-1]
    frame = ff.getFingertips(frame, cnt)
    cv2.drawContours(frame, biggestContour, -1, (0, 255, 0), 3)

    cv2.imshow('frame', dst)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
