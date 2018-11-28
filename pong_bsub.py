import numpy as np
import cv2
import filter_frame

def getMahalanobisImage(input, mean, std, threshold):
    res = ((input - mean)**2) / (std**2)
    return cv2.threshold(res, threshold**2, 255, cv2.THRESH_BINARY)[1]

if __name__ == "__main__":
    camera = cv2.VideoCapture(0)

    ret, frame = camera.read()
    h, w = frame.shape[:2]

    top, bottom, left, right = 0, h, 0, int(w / 2)
    set_background_period = 30

    background_frames = np.empty([h, int(w / 2), set_background_period])

    for i in range(set_background_period):
        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[top:bottom, left:right]
        background_frames[:,:,i] = grayscale

    mean = np.mean(background_frames, axis=2)
    std = np.std(background_frames, axis=2)
    std[std < 0.001] = 0.001
    threshold = 6
    ff = filter_frame.FilterFrame()

    while True:
        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)

        original = frame.copy()
        player_half = frame[top:bottom, left:right,:]
        grayscale = cv2.cvtColor(player_half, cv2.COLOR_BGR2GRAY)
        grayscale = cv2.GaussianBlur(grayscale, (7,7), 0)

        bg_sub = getMahalanobisImage(grayscale, mean, std, threshold).astype('uint8')

        _, contours, _ = cv2.findContours(bg_sub, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            sorted_contours = np.array(sorted(contours, key=cv2.contourArea))
            biggest_contour = sorted_contours[-1:]
            cnt = sorted_contours[-1]
            
            moment = cv2.moments(cnt)

            if moment['m00'] != 0:
                cx = int(moment['m10'] / moment['m00'])
                cy = int(moment['m01'] / moment['m00'])
                cv2.circle(original, (cx, cy), 5, [255, 0, 0], -1)

            frame, paddle_point = ff.getFingertips(original, cnt)

            cv2.drawContours(frame, biggest_contour, -1, (0, 255, 0), 3)

        cv2.imshow('bg_sub', bg_sub)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
