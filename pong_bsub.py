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
    return abs(first[0] - second[0]) / (width * 0.01)

def getMahalanobisImage(r,b,g, mean_r,mean_b,mean_g, std_r,std_b,std_g, threshold):
    
    res_r = ((r - mean_r)**2) / (std_r**2)
    res_b = ((b - mean_b)**2) / (std_b**2)
    res_g = ((g - mean_g)**2) / (std_g**2)

    
    t_r = cv2.threshold(res_r, threshold**2, 255, cv2.THRESH_BINARY)[1]
    
    t_b = cv2.threshold(res_b, threshold**2, 255, cv2.THRESH_BINARY)[1]
    t_g = cv2.threshold(res_g, threshold**2, 255, cv2.THRESH_BINARY)[1]


    print(cv2.bitwise_or(t_r,t_b,t_g))

    return cv2.bitwise_and(t_r,t_b,t_g)


if __name__ == "__main__":
    camera = cv2.VideoCapture(0)

    ret, frame = camera.read()
    height, width = frame.shape[:2]

    top, bottom, left, right = 0, height, 0, int(width / 2)
    set_background_period = 30

    background_frames_b = np.empty([height, int(width / 2), set_background_period])
    background_frames_g = np.empty([height, int(width / 2), set_background_period])


    background_frames_r = np.empty([height, int(width / 2), set_background_period])


    for i in range(set_background_period):
        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)
        b,g,r = cv2.split(frame[top:bottom, left:right,:])
        

        
    


        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[top:bottom, left:right]
        grayscale = cv2.GaussianBlur(grayscale, (7,7), 0)
        background_frames_r[:,:,i] = r
        background_frames_g[:,:,i] = g
        background_frames_b[:,:,i] = b


    mean_r= np.mean(background_frames_r, axis=2)
    mean_g = np.mean(background_frames_g, axis=2)
    mean_b = np.mean(background_frames_b, axis=2)
    
    std_r = np.std(background_frames_r, axis=2)
    std_b = np.std(background_frames_b, axis=2)
    std_g = np.std(background_frames_g, axis=2)
    


        
           
    

    
   
    #std[std < 0.00001] = 0.00001
    threshold = 3
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
    
    out = cv2.VideoWriter('output_1.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10.0, (height,width))    
    out1 = cv2.VideoWriter('output_2.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10.0, (height,width))

    
    
    while True:
        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)

        original = frame.copy()
        player_half = frame[top:bottom, left:right,:]
        grayscale = cv2.cvtColor(player_half, cv2.COLOR_BGR2GRAY)
        grayscale = cv2.GaussianBlur(grayscale, (7,7), 0)

        b,g,r = cv2.split(frame[top:bottom, left:right,:])



        bg_sub = getMahalanobisImage(r,b,g, mean_r,mean_b,mean_g, std_r,std_b,std_g, threshold).astype('uint8')

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


        out.write(frame)
        out1.write(bg_sub)
        
        cv2.imshow('bg_sub', bg_sub)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or ended:
            break

    out.close()
    out1.close()
    camera.release()
    cv2.destroyAllWindows()
