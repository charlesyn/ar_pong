import numpy as np
import cv2
import filter_frame
import game_board
import math

def get_velocity_direction(prevIm, currIm, oldPoints):
    newPoints, status, err = cv2.calcOpticalFlowPyrLK(prevIm, currIm, oldPoints, None)
    x1, y1 = oldPoints[0].ravel()
    x2, y2 = newPoints[0].ravel()
    if math.abs(x2 - x1) < 0.01:
        x2 += 0.02
    dist = np.linalg.norm(newPoints[0]-oldPoints[0])
    deg = np.arctan((y2-y1)/(x2-x1))*180/np.pi
    return dist, deg

def getAngle(wrist, finger):
    r = wrist[0]-finger[0]
    c = wrist[1]-finger[1]
    return c, r

def get_speed(first, second, width):
    return abs(first[0] - second[0]) / (width * 0.1)

def getMahalanobisImage(input, mean, std, threshold):
    res = ((input - mean)**2) / (std**2)
    return cv2.threshold(res, threshold**2, 255, cv2.THRESH_BINARY)[1]

if __name__ == "__main__":
    camera = cv2.VideoCapture(0)

    ret, frame = camera.read()
    height, width = frame.shape[:2]

    top, bottom, left, right = 0, height, 0, int(width / 2)
    set_background_period = 30

    background_frames = np.empty([height, int(width / 2), set_background_period])

    for i in range(set_background_period):
        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[top:bottom, left:right]
        grayscale = cv2.GaussianBlur(grayscale, (7,7), 0)
        background_frames[:,:,i] = grayscale

    mean = np.mean(background_frames, axis=2)
    std = np.std(background_frames, axis=2)
    std[std < 0.00001] = 0.00001
    threshold = 5
    ff = filter_frame.FilterFrame()
    pong = game_board.Pong(
        h=height,
        w=width,
        default_ball_dx=width//100,
        default_ball_dy=width//100,
        default_paddle_speed=height//100,
        default_half_paddle_height=height//10)
    first = np.array([0, 0])
    second = np.array([0, 0])
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
            pong.set_cx(paddle_point[0])
            pong.set_cy(paddle_point[1])
            first = second
            second = paddle_point
            moment = cv2.moments(cnt)
            if moment['m00'] != 0:
                cx = int(moment['m10'] / moment['m00'])
                cy = int(moment['m01'] / moment['m00'])
                cv2.circle(original, (cx, cy), 5, [255, 0, 0], -1)
            cv2.drawContours(frame, biggest_contour, -1, (0, 255, 0), 3)

        speed = get_speed(first, second, width)
        ended = pong.update(speed)
        pong.draw(frame)

        cv2.imshow('bg_sub', bg_sub)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or ended:
            break

    camera.release()
    cv2.destroyAllWindows()
