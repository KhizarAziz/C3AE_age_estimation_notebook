import os
import cv2
import numpy as np
import math


def quaterniondToRulerAngle(quaterniond):
    q = quaterniond
    y_sqrt = q.y ** 2
    # pitch
    t0 = 2 * (q.w * q.x + q.y * q.z)
    t1 = 1.0 - 2.0 * (q.x ** 2 + y_sqrt) 
    pitch = math.atan(t0 / t1)#math.atan2(t0, t1)

    # yaw
    t2 = 2 * (q.w * q.y - q.z * q.x)
    t2 = max(min(t2, 1), -1)
    yaw = math.asin(t2)

    # roll
    t3 = 2 * (q.w * q.z + q.x * q.y)
    t4 = 1 - 2 * (y_sqrt + q.z * q.z)
    roll = math.atan(t3 / t4)  #math.atan2(t3, t4)
    return pitch, yaw, roll

def tran_euler(rotation_vect):
    theta = cv2.norm(rotation_vect, cv2.NORM_L2)
    class Quation(object):
        def __init__(self, w, x, y, z):
            self.x = x
            self.y = y
            self.z = z
            self.w = w
    quat = Quation(
        math.cos(theta / 2),
        math.sin(theta / 2) * rotation_vect[0][0] / theta,
        math.sin(theta / 2) * rotation_vect[1][0] / theta,
        math.sin(theta / 2) * rotation_vect[2][0] / theta
        )
    return map(lambda x: x / math.pi * 180, quaterniondToRulerAngle(quat))

def trans_landmarks(img, landmark_groups):
    result = []
    for lm in landmark_groups:
        landmarks = np.array([(lm[x], lm[5 + x],) for x in range(5)], dtype="double")
        for p in landmarks:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
        result.append(get_rotation_angle(img, landmarks))
    return result

def get_rotation_angle(img, landmarks, draw=False):

    # you can read more about this method model on https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/

    size = img.shape
    parts = landmarks.parts()

    #making an array of points(nose,chin,left-eye etc) from lm, which will be used to determine face angle
    image_points = np.array([
                                (parts[30].x, parts[30].y),     # Nose tip : point 30 in landmarks list
                                (parts[8].x, parts[8].y),     # Chin : same
                                (parts[36].x, parts[36].y),     # Left eye left corner
                                (parts[45].x, parts[45].y),     # Right eye right corne
                                (parts[48].x, parts[48].y),     # Left Mouth corner
                                (parts[54].x, parts[54].y)      # Right mouth corner
                            ], dtype="double")

    #3D locations of the same points : You also need the 3D location of the 2D feature points. You might be thinking that you need a 3D model of the person in the photo to get the 3D locations. Ideally yes, but in practice, you don’t. A generic 3D model will suffice. Where do you get a 3D model of a head from ? Well, you really don’t need a full 3D model. You just need the 3D locations of a few points in some arbitrary reference frame. In this tutorial, we are going to use the following 3D points.
    model_points = np.array([
              (0.0, 0.0, 0.0),             # Nose tip
              (0.0, -330.0, -65.0),        # Chin
              (-225.0, 170.0, -135.0),     # Left eye left corner
              (225.0, 170.0, -135.0),      # Right eye right corne
              (-150.0, -150.0, -125.0),    # Left Mouth corner
              (150.0, -150.0, -125.0)      # Right mouth corner
          ])

    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2,)
    camera_matrix = np.array([
             [focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]
         ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))
    (success, rotation_vector, trans_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
    f_pitch, f_yaw, f_roll = tran_euler(rotation_vector)

    n_pitch = prod_trans_point((0, 0, 500.0), rotation_vector, trans_vector, camera_matrix, dist_coeffs)
    n_yaw = prod_trans_point((200.0, 0, 0), rotation_vector, trans_vector, camera_matrix, dist_coeffs)
    n_roll = prod_trans_point((0, 500.0, 0), rotation_vector, trans_vector, camera_matrix, dist_coeffs)

    if draw:
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        cv2.line(img, p1, n_roll, (255, 0, 0), 2)
        cv2.line(img, p1, n_yaw, (0, 255, 0), 2)
        cv2.line(img, p1, n_pitch, (0, 0, 255), 2)
        cv2.putText(img, ("r:" + str(f_roll))[:6], n_roll, 1,1, (136, 97, 45), 2) 
        cv2.putText(img, ("y:" + str(f_yaw))[:6], n_yaw, 1,1, (136, 97, 45), 2) 
        cv2.putText(img, ("p:" + str(f_pitch))[:6], n_pitch, 1,1, (136, 97, 45), 2) 
    return f_pitch, f_yaw, f_roll

def prod_trans_point(p3d, rotation_vector, trans_vector, camera_matrix, dist_coeffs):
    plane_point, _ = cv2.projectPoints(np.array([p3d]), rotation_vector, trans_vector, camera_matrix, dist_coeffs)
    return (int(plane_point[0][0][0]), int(plane_point[0][0][1]))

