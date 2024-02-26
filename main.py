import cv2
import numpy as np


def rbf_kernel(A, a, h):
    c = A-a
    c = c.reshape(-1)
    k = c.dot(c)
    return np.exp(-k/(2*h*h))

def meanshift(img,obj,sigma,h,y_pos,its =0):
    height, weight = obj.shape[:2]
    y_leftup = np.array([y_pos[0]-height//2, y_pos[1]-weight//2])
    roi_y = img[y_pos[0]-height//2:y_pos[0] +
                height//2, y_pos[1]-weight//2:y_pos[1]+weight//2]
    xc = np.array([obj.shape[0]//2, obj.shape[1]//2])
    sum1 = 0
    sum2 = 0
    
    x_pts = np.meshgrid(np.arange(weight),np.arange(height))
    omega = (x_pts[0]-xc[1])**2+(x_pts[1]-xc[0])**2
    omega = np.exp(-omega/(2*sigma*sigma))
    y_pts = [x_pts[0]+y_leftup[1],x_pts[1]+y_leftup[0]]
    g = (y_pts[0]-y_pos[1])**2+(y_pts[1]-y_pos[0])**2
    g = np.exp(-g/(2*sigma*sigma))
    for i in range(height):
        for j in range(weight):
            k = (obj[i,j,:]-roi_y)**2
            k = k.sum(axis=2).astype(np.float64)
            k = np.exp(-k/(2*h*h))
            sum1 = sum1 + omega[i,j]*np.sum(np.sum(g*k*y_pts,axis=1),axis=1)
            sum2 = sum2 + omega[i,j]*np.sum(np.sum(g*k))
    y_newpos = np.round(sum1/sum2).astype(np.int16)[::-1]
    if np.linalg.norm(y_newpos-y_pos) < 0.5 or its >= 10:
        return y_newpos
    else: 
        y_newpos = meanshift(img, obj, sigma, h, y_newpos, its+1)
        return y_newpos
    
    


if __name__ == '__main__':
    video = cv2.VideoCapture('1.avi')

    # 第一帧截取目标
    ret, frame = video.read()
   
    # 选择ROI
    roi = cv2.selectROI(frame)
    y_pos = np.array([roi[1]+roi[3]//2, roi[0]+roi[2]//2])
    roi = frame[int(roi[1]):int(roi[1]+roi[3]//2*2),
                int(roi[0]):int(roi[0]+roi[2]//2*2)]
    cv2.imshow('roi', roi)
    n = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if n == 0:
            n = n+1
            continue
        y_pos = meanshift(frame, roi, 10, 11, y_pos)
        # 画矩形框
        cv2.rectangle(frame, (int(y_pos[1]-roi.shape[1]//2), int(y_pos[0]-roi.shape[0]//2)),
                      (int(y_pos[1]+roi.shape[1]//2), int(y_pos[0]+roi.shape[0]//2)), (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
