# -*- coding:utf-8 -*-
'''
#Author :Administrator
#Time   :2019/9/7 16:32
#File   :faceChange.py
#IDE    :PyCharm
'''


import dlib
import cv2
import os
import numpy as np
import pandas as pd
import math
import glob
import time
"""
功能：FaceDescriptor 能将图片抽取68个特征点坐标，以及用128维向量来描述图片
"""
IMAGES_DEV='H:/cmq/securityAI_round1_dev.csv'
IMAGES_PATH='H:/cmq/securityAI_round1_images/images/'

OUTPUT_PATH='H:/cmq/securityAI_round1_images/outputs/deadline/images/'

current_path = os.getcwd()  # 获取当前路径

print(current_path)

VECTOR_ALL_PATH=current_path+"/data/vector_all_1.csv"

DISTANCE_ALL_PATH=current_path+"/data/distance_all.csv"

#模型路径
predictor_path =current_path + "/model/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = current_path + "/model/dlib_face_recognition_resnet_model_v1.dat"

# 读入模型
detector = dlib.get_frontal_face_detector()   #加载dlib自带的人脸检测器 定位出人脸
shape_predictor = dlib.shape_predictor(predictor_path)    # 选取人脸68个特征点检测器
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path) #将人脸用128维向量表示出来


class TooManyFaces(Exception):
    pass

class NoFace(Exception):
    pass

class FaceChangerAll(object):

    def __init__(self, which_predictor='68'):
        # print('Starting your FaceChanger...')

        self.current_path = os.getcwd()
        # print('Current path:{0}'.format(self.current_path))

        predictor_68_points_path = self.current_path + '/model/shape_predictor_68_face_landmarks.dat'
        predictor_5_points_path = self.current_path + '/model/shape_predictor_5_face_landmarks.dat'
        if which_predictor == '68':
            predictor_name = 'shape_predictor_68_face_landmarks.dat'
            self.predictor_path = predictor_68_points_path
        elif which_predictor == '5':
            predictor_name = 'shape_predictor_5_face_landmarks.dat'
            self.predictor_path = predictor_5_points_path
        else:
            predictor_name = 'shape_predictor_68_face_landmarks.dat'
            self.predictor_path = predictor_68_points_path
        # print('Your predictor is:{0}'.format(predictor_name))


        # some parameters
        self.SCALE_FACTOR = 1
        self.FEATHER_AMOUNT = 11

        self.FACE_POINTS = list(range(17, 68))  #轮廓
        self.MOUTH_POINTS = list(range(48, 61)) #嘴巴
        self.RIGHT_BROW_POINTS = list(range(17, 22))
        self.LEFT_BROW_POINTS = list(range(22, 27))
        self.RIGHT_EYE_POINTS = list(range(36, 42))
        self.LEFT_EYE_POINTS = list(range(42, 48))
        self.NOSE_POINTS = list(range(27, 35)) #鼻子
        self.JAW_POINTS = list(range(0, 17)) #下巴

        # Points used to line up the images.
        self.ALIGN_POINTS = (self.LEFT_BROW_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_EYE_POINTS +
                             self.RIGHT_BROW_POINTS + self.NOSE_POINTS + self.MOUTH_POINTS+
                             self.JAW_POINTS+self.FACE_POINTS)

        self.OVERLAY_POINTS = [
            self.LEFT_BROW_POINTS +self.RIGHT_BROW_POINTS +
            self.LEFT_EYE_POINTS + self.RIGHT_EYE_POINTS+
            self.NOSE_POINTS+self.MOUTH_POINTS
        ]

        self.COLOUR_CORRECT_BLUR_FRAC = 0.6

        # load in models
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)

        self.image1 = None
        self.image2 = None
        self.landmarks1 = None
        self.landmarks2 = None

    def load_images(self, image1_path, image2_path):

        self.image1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
        self.image2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)

        self.landmarks1 = self.get_landmark(self.image1)
        self.landmarks2 = self.get_landmark(self.image2)

    def run(self, showProcedure=False, saveResult=True,output_path=None):
        if self.image1 is None or self.image2 is None:
            print('You need to load two images first.')
            return

        if showProcedure == True:
            print('Showing the procedure.Press any key to continue your process.')
            cv2.imshow("1", self.image1)
            cv2.waitKey(0)
            cv2.imshow("2", self.image2)
            cv2.waitKey(0)

        M = self.transformation_from_points(self.landmarks1[self.ALIGN_POINTS], self.landmarks2[self.ALIGN_POINTS])

        mask = self.get_face_mask(self.image2, self.landmarks2)
        if showProcedure == True:
            cv2.imshow("3", mask)
            cv2.waitKey(0)

        warped_mask = self.warp_image(mask, M, self.image1.shape)
        if showProcedure == True:
            cv2.imshow("4", warped_mask)
            cv2.waitKey(0)

        combined_mask = np.max([self.get_face_mask(self.image1, self.landmarks1), \
                                warped_mask], axis=0)
        if showProcedure == True:
            cv2.imshow("5", combined_mask)
            cv2.waitKey(0)

        warped_img2 = self.warp_image(self.image2, M, self.image1.shape)
        if showProcedure == True:
            cv2.imshow("6", warped_img2)
            cv2.waitKey(0)

        warped_corrected_img2 = self.correct_colours(self.image1, warped_img2, self.landmarks1)
        warped_corrected_img2_temp = np.zeros(warped_corrected_img2.shape, dtype=warped_corrected_img2.dtype)
        cv2.normalize(warped_corrected_img2, warped_corrected_img2_temp, 0, 1, cv2.NORM_MINMAX)
        if showProcedure == True:
            cv2.imshow("7", warped_corrected_img2_temp)
            cv2.waitKey(0)

        output = self.image1 * (1.0 - combined_mask) + warped_corrected_img2 * combined_mask
        output_show = np.zeros(output.shape, dtype=output.dtype)
        cv2.normalize(output, output_show, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(output, output, 0, 255, cv2.NORM_MINMAX)

        if showProcedure == True and output_path is not None:
            cv2.imshow("8", output_show.astype(output_show.dtype))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if saveResult is True:
            cv2.imwrite(output_path, output)

    def get_landmark(self, image):
        face_rect = self.detector(image, 1)

        if len(face_rect) > 1:
            print('Too many faces.We only need no more than one face.')
            raise TooManyFaces
        elif len(face_rect) == 0:
            print('No face.We need at least one face.')
            raise NoFace
        else:
            return np.matrix([[p.x, p.y] for p in self.predictor(image, face_rect[0]).parts()])

    def transformation_from_points(self, points1, points2):
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)

        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2

        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2

        U, S, Vt = np.linalg.svd(points1.T * points2)
        R = (U * Vt).T

        return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])

    def warp_image(self, image, M, dshape):
        output_image = np.zeros(dshape, dtype=image.dtype)
        cv2.warpAffine(image, M[:2], (dshape[1], dshape[0]), dst=output_image, flags=cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_TRANSPARENT)
        return output_image

    def correct_colours(self, im1, im2, landmarks1):
        blur_amount = self.COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
            np.mean(landmarks1[self.LEFT_EYE_POINTS], axis=0) -
            np.mean(landmarks1[self.RIGHT_EYE_POINTS], axis=0))
        blur_amount = int(blur_amount)
        if blur_amount % 2 == 0:
            blur_amount += 1
        im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
        im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

        # Avoid divide-by-zero errors.
        im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

        return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                im2_blur.astype(np.float64))

    def draw_convex_hull(self, img, points, color):
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(img, points, color)

    def get_face_mask(self, img, landmarks):
        img = np.zeros(img.shape[:2], dtype=np.float64)
        for group in self.OVERLAY_POINTS:
            self.draw_convex_hull(img, landmarks[group], color=1)

        img = np.array([img, img, img]).transpose((1, 2, 0))

        img = (cv2.GaussianBlur(img, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0) > 0) * 1.0
        img = cv2.GaussianBlur(img, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0)

        return img

class FaceChanger(object):
    def __init__(self, which_predictor='68'):
        # print('Starting your FaceChanger...')

        self.current_path = os.getcwd()
        # print('Current path:{0}'.format(self.current_path))

        predictor_68_points_path = self.current_path + '/model/shape_predictor_68_face_landmarks.dat'
        predictor_5_points_path = self.current_path + '/model/shape_predictor_5_face_landmarks.dat'
        if which_predictor == '68':
            predictor_name = 'shape_predictor_68_face_landmarks.dat'
            self.predictor_path = predictor_68_points_path
        elif which_predictor == '5':
            predictor_name = 'shape_predictor_5_face_landmarks.dat'
            self.predictor_path = predictor_5_points_path
        else:
            predictor_name = 'shape_predictor_68_face_landmarks.dat'
            self.predictor_path = predictor_68_points_path
        # print('Your predictor is:{0}'.format(predictor_name))

        # some parameters
        self.SCALE_FACTOR = 1
        self.FEATHER_AMOUNT = 11

        self.FACE_POINTS = list(range(17, 68))
        self.MOUTH_POINTS = list(range(48, 61))
        self.RIGHT_BROW_POINTS = list(range(17, 22))
        self.LEFT_BROW_POINTS = list(range(22, 27))
        self.RIGHT_EYE_POINTS = list(range(36, 42))
        self.LEFT_EYE_POINTS = list(range(42, 48))
        self.NOSE_POINTS = list(range(27, 35))  # 鼻子
        self.JAW_POINTS = list(range(0, 17))  # 下巴

        # Points used to line up the images.
        self.ALIGN_POINTS = (self.LEFT_BROW_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_EYE_POINTS +
                             self.RIGHT_BROW_POINTS + self.NOSE_POINTS + self.MOUTH_POINTS +
                             self.JAW_POINTS + self.FACE_POINTS)

        self.OVERLAY_POINTS_1 = [
            self.LEFT_EYE_POINTS
        ]
        self.OVERLAY_POINTS_2 = [
            self.RIGHT_EYE_POINTS
        ]
        self.OVERLAY_POINTS_3 = [
            self.NOSE_POINTS
        ]
        self.OVERLAY_POINTS_4 = [
            self.MOUTH_POINTS
        ]
        self.OVERLAY_POINTS_5 = [
            self.LEFT_BROW_POINTS
        ]
        self.OVERLAY_POINTS_6 = [
            self.RIGHT_BROW_POINTS
        ]


        self.COLOUR_CORRECT_BLUR_FRAC = 0.6

        # load in models
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)

        self.image1 = None
        self.image2 = None
        self.landmarks1 = None
        self.landmarks2 = None

    def load_images(self, image1_path, image2_path):

        self.image1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
        self.image2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)

        self.landmarks1 = self.get_landmark(self.image1)
        self.landmarks2 = self.get_landmark(self.image2)

    def run(self, showProcedure=False, saveResult=True, output_path=None):
        if self.image1 is None or self.image2 is None:
            print('You need to load two images first.')
            return

        if showProcedure == True:
            print('Showing the procedure.Press any key to continue your process.')
            cv2.imshow("1", self.image1)
            cv2.waitKey(0)
            cv2.imshow("2", self.image2)
            cv2.waitKey(0)

        M = self.transformation_from_points(self.landmarks1[self.ALIGN_POINTS], self.landmarks2[self.ALIGN_POINTS])

        mask = self.get_face_mask(self.image2, self.landmarks2)  # 得到图片2掩膜

        if showProcedure == True:
            cv2.imshow("3", mask)
            cv2.waitKey(0)

        warped_mask = self.warp_image(mask, M, self.image1.shape)
        if showProcedure == True:
            cv2.imshow("4", warped_mask)
            cv2.waitKey(0)

        combined_mask = np.max([self.get_face_mask(self.image1, self.landmarks1), \
                                warped_mask], axis=0)
        if showProcedure == True:
            cv2.imshow("5", combined_mask)
            cv2.waitKey(0)

        warped_img2 = self.warp_image(self.image2, M, self.image1.shape)
        if showProcedure == True:
            cv2.imshow("6", warped_img2)
            cv2.waitKey(0)

        warped_corrected_img2 = self.correct_colours(self.image1, warped_img2, self.landmarks1)
        warped_corrected_img2_temp = np.zeros(warped_corrected_img2.shape, dtype=warped_corrected_img2.dtype)

        cv2.normalize(warped_corrected_img2, warped_corrected_img2_temp, 0, 1, cv2.NORM_MINMAX)
        if showProcedure == True:
            cv2.imshow("7", warped_corrected_img2_temp)
            cv2.waitKey(0)

        output = self.image1 * (1.0 - combined_mask) + warped_corrected_img2 * combined_mask
        output_show = np.zeros(output.shape, dtype=output.dtype)
        cv2.normalize(output, output_show, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(output, output, 0, 255, cv2.NORM_MINMAX)

        if showProcedure == True and output_path is not None:
            cv2.imshow("8", output_show.astype(output_show.dtype))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if saveResult is True:
            cv2.imwrite(output_path, output)

    def get_landmark(self, image):
        face_rect = self.detector(image, 1)

        if len(face_rect) > 1:
            print('Too many faces.We only need no more than one face.')
            raise TooManyFaces
        elif len(face_rect) == 0:
            print('No face.We need at least one face.')
            raise NoFace
        else:
            # print('left {0}; top {1}; right {2}; bottom {3}'.format(face_rect[0].left(), face_rect[0].top(),
            #                                                         face_rect[0].right(), face_rect[0].bottom()))
            return np.matrix([[p.x, p.y] for p in self.predictor(image, face_rect[0]).parts()])

    def transformation_from_points(self, points1, points2):
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)

        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2

        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2

        U, S, Vt = np.linalg.svd(points1.T * points2)
        R = (U * Vt).T

        return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])

    def warp_image(self, image, M, dshape):
        output_image = np.zeros(dshape, dtype=image.dtype)
        cv2.warpAffine(image, M[:2], (dshape[1], dshape[0]), dst=output_image, flags=cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_TRANSPARENT)
        return output_image

    def correct_colours(self, im1, im2, landmarks1):
        blur_amount = self.COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
            np.mean(landmarks1[self.LEFT_EYE_POINTS], axis=0) -
            np.mean(landmarks1[self.RIGHT_EYE_POINTS], axis=0))
        blur_amount = int(blur_amount)
        if blur_amount % 2 == 0:
            blur_amount += 1
        im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
        im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

        # Avoid divide-by-zero errors.
        im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

        return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                im2_blur.astype(np.float64))

    def draw_convex_hull(self, img, points, color):
        area1 = cv2.convexHull(points)  # 根据坐标值计算凸包
        # area2 = cv2.convexHull(points)  # 根据坐标值计算凸包

        cv2.fillConvexPoly(img, area1, color)

    def get_face_mask(self, img, landmarks):
        """
        得到图片2的掩模
        :param img:  self.image2 bgr
        :param landmarks: 图片的关键点 68个关键点 矩阵形式
        :return:
        """
        img = np.zeros(img.shape[:2], dtype=np.float64)

        for group1, group2, group3, group4, group5, group6 in zip(self.OVERLAY_POINTS_1, self.OVERLAY_POINTS_2,
                                                                  self.OVERLAY_POINTS_3, self.OVERLAY_POINTS_4,
                                                                  self.OVERLAY_POINTS_5, self.OVERLAY_POINTS_6):
            # print('group1:', group1)
            # print('group2:', group2)
            self.draw_convex_hull(img, landmarks[group1], color=1)
            self.draw_convex_hull(img, landmarks[group2], color=1)
            self.draw_convex_hull(img, landmarks[group3], color=1)
            self.draw_convex_hull(img, landmarks[group4], color=1)
            self.draw_convex_hull(img, landmarks[group5], color=1)
            self.draw_convex_hull(img, landmarks[group6], color=1)

        img = np.array([img, img, img]).transpose((1, 2, 0))
        # print('img:', img)

        img = (cv2.GaussianBlur(img, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0) > 0) * 1.0
        img = cv2.GaussianBlur(img, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0)

        return img

def get_image_names(path):
    """
    #从csv文件中读取图片的名称
    :param path:
    :return:
    """
    dev=pd.read_csv(path)

    dev=np.array(dev)

    print('dev:{}'.format(dev.shape))
    imageName=dev[:,1:2]
    return imageName

def get_68_landmark(image_path):
    """
    求每张图片的68个关键点坐标
    :param image: cv2读取的文件
    :return: [[x,y],[x,y]...[x,y]] len=68 list格式
    """
    img_bgr = cv2.imread(image_path)

    face_rect = detector(img_bgr, 1)

    if len(face_rect) > 1:
        print('Too many faces.We only need no more than one face.')
        raise TooManyFaces
    elif len(face_rect) == 0:
        print('No face.We need at least one face.')
        raise NoFace
    else:
        print('left {0}; top {1}; right {2}; bottom {3}'.format(face_rect[0].left(), face_rect[0].top(),
                                                                face_rect[0].right(), face_rect[0].bottom()))

    points = np.matrix([[p.x, p.y] for p in shape_predictor(img_bgr, face_rect[0]).parts()])

    points = points.tolist()
    return points

def get_128vector(img_path):
    """
    根据图片路径得到128维向量
    :param img_path:
    :return: li :list
    """
    vec_path_start_time=time.time()

    img_bgr = cv2.imread(img_path)
    b, g, r = cv2.split(img_bgr)
    img_rgb = cv2.merge([r, g, b])

    face_rect = detector(img_bgr, 1)

    if len(face_rect) > 1:
        print('Too many faces.We only need no more than one face.')
        raise TooManyFaces
    elif len(face_rect) == 0:
        print('No face.We need at least one face.')
        raise NoFace
    else:

        shape = shape_predictor(img_bgr, face_rect[0])
        face_descriptor = face_rec_model.compute_face_descriptor(img_rgb, shape)

    li = []
    for index, ve in enumerate(face_descriptor):
        li.append(ve)

    vec_path_end_time = time.time()
    print('get vector from image_path need time:{}'.format(vec_path_end_time - vec_path_start_time))

    return li

def compareTwoPerson(vector_a, vector_b):
    """
    计算两个图片之间的欧式距离
    :param per1: 第一个图片128list
    :param per2: 第二张图片128list
    :return:
    """
    diff = 0
    for i in range(len(vector_a)):
        diff += (vector_a[i] - vector_b[i]) ** 2
    diff = np.sqrt(diff)
    # print('diff:',diff)
    return diff

def getTwoPicDiffC(img1_path,img2_path):
    """
    得到两个图片的差值
    :param img1_path:
    :param img2_path:
    :return:
    """
    C_start_tiem=time.time()

    img1_bgr = cv2.imread(img1_path)
    b_1, g_1, r_1 = cv2.split(img1_bgr)

    img2_bgr = cv2.imread(img2_path)
    b_2, g_2, r_2 = cv2.split(img2_bgr)



    r_1=np.array(r_1,dtype=np.int16)
    r_2=np.array(r_2,dtype=np.int16)

    g_1 = np.array(g_1, dtype=np.int16)
    g_2 = np.array(g_2, dtype=np.int16)

    b_1 = np.array(b_1, dtype=np.int16)
    b_2 = np.array(b_2, dtype=np.int16)

    # print('A:')
    # print('r:')
    # print(r_1)
    #
    # print('B:')
    # print('r:')
    # print(r_2)


    r_diff = r_2 - r_1
    g_diff = g_2 - g_1
    b_diff = b_2 - b_1


    # print('C:')
    # print('r_diff:')
    # print(r_diff)

    C_end_tiem=time.time()
    # print('getC need time:{}'.format(C_end_tiem-C_start_tiem))
    return r_diff,g_diff,b_diff

def getDiffConstrainsD(r_diff,g_diff,b_diff):
    """
    将C=B-A 中C的像素值约束一下
    :param r_diff:
    :param g_diff:
    :param b_diff:
    :return:
    """
    D_start_time=time.time()

    # print('fangfa2:') #耗时更短
    r_diff[r_diff > 25.5 ]= 25.5
    r_diff[r_diff < -25.5] = -25.5

    g_diff[g_diff > 25.5] = 25.5
    g_diff[g_diff < -25.5] = -25.5

    b_diff[b_diff > 25.5] = 25.5
    b_diff[b_diff < -25.5] = -25.5

    # print('D:')
    # print('r_diff:')
    # print(r_diff)

    D_end_time = time.time()
    # print('getD need time:{}'.format(D_end_time-D_start_time))

    return r_diff,g_diff,b_diff

def getConstraintedPicE(imgA_path,imgB_path):
    """
    :param img1_path: 原始图片
    :param img2_path: 待替换图片
    :return:
    """
    E_start_time=time.time()
    imgA_bgr = cv2.imread(imgA_path)
    b_A, g_A, r_A = cv2.split(imgA_bgr)

    r_diff, g_diff, b_diff=getTwoPicDiffC(imgA_path,imgB_path) #C=B-A
    r_diff, g_diff, b_diff=getDiffConstrainsD(r_diff,g_diff,b_diff) #D=F(C)

    #E=A+D 对图片E分别求3通道
    r_E=r_A+r_diff
    g_E=g_A+g_diff
    b_E=b_A+b_diff

    r_E = np.array(r_E, dtype=np.uint8)
    g_E = np.array(g_E, dtype=np.uint8)
    b_E = np.array(b_E, dtype=np.uint8)

    #把图片存起来
    imgE_rgb = cv2.merge([b_E, g_E,r_E])

    E_end_time = time.time()
    # print('getE need time:{}'.format(E_end_time-E_start_time))
    return imgE_rgb


def get_dist_AE(vector_A,img_bgr_E):
    """
    会出现识别不出人脸的问题，在这里加上一个标签
    :param vector_A:
    :param img_bgr_E:
    :return: dist
    """
    distAE_start_time=time.time()

    #是否出现检测不到人脸的情况
    NoDetectedFace=False      #False表示检测到人脸 True表示出现检测不到人脸的状况

    b, g, r = cv2.split(img_bgr_E)
    img_rgb_E = cv2.merge([r, g, b])
    face_rect = detector(img_bgr_E, 1)


    if len(face_rect) > 1:
        print('Too many faces.We only need no more than one face.')
        NoDetectedFace=True #检测到多张人脸也作为检测不到人脸来处理
    elif len(face_rect) == 0:
        print('No face.We need at least one face.')
        NoDetectedFace=True  #出现检测不到人脸情况


    if NoDetectedFace==False:
        vector_E = []
        #检测到人脸情况
        shape = shape_predictor(img_bgr_E, face_rect[0])
        face_descriptor = face_rec_model.compute_face_descriptor(img_rgb_E, shape)

        #将向量转为128list形式
        for index, ve in enumerate(face_descriptor):
            vector_E.append(ve)
    else:vector_E=vector_A
    #检测不到人脸，那就用A的向量

    dist=compareTwoPerson(vector_A,vector_E)
    # print('A E dist:',dist)

    distAE_end_time = time.time()
    # print('distAE need time:{}'.format(distAE_end_time-distAE_start_time))

    return dist


def get_distance_all(imageName):
    """
    得到所有的距离，并保存起来，那些检测不到人脸的E，距离用0.0代替
    :param imageName:
    :return:
    """
    pic_number=len(imageName)
    print('pic number:{}'.format(pic_number))

    dist_all=[]
    for i in range(pic_number):
        pic_start_time = time.time()

        img1_path=IMAGES_PATH+imageName[i]
        img1_path = img1_path[0]
        vector_1=get_128vector(img1_path)

        print('pic={},path:{}'.format(i+1,img1_path))

        dist_pic=[]

        #遍历所有图片找到图片的E
        for j in range(pic_number):
            img2_path=IMAGES_PATH+imageName[j]
            img2_path=img2_path[0]

            imageE=getConstraintedPicE(img1_path,img2_path)
            #注意：这里存在图片E识别不出人脸的问题 需要改一下代码，直接求距离，根据A的vector 和E这张图片img_bgr
            dist=get_dist_AE(vector_1,imageE)

            print('dist of pic {},pic {} is {}'.format(imageName[i], imageName[j], dist))

            dist_pic.append(dist)
        dist_all.append(dist_pic)

        pic_end_time = time.time()
        print('get pic:{} all:{} distance need time={}'.format(imageName[i], pic_number, pic_end_time - pic_start_time))

    #存起来
    dist_all_np=np.array(dist_all)
    dist_all_pd=pd.DataFrame(dist_all_np)
    dist_all_pd.to_csv(path_or_buf=current_path+'/data/distance_all_1.csv', header=False, index=False)

def getindex(distance_all):
    """
    得到需要的索引值，返回173张图片的索引
    :param distance_all:
    :return:
    """
    max_index_all = np.argmax(distance_all, axis=1)  # 每一行的最大索引 173个数
    index_all=[]
    pic_num=len(distance_all)

    for  i in range(pic_num):
        #计算一个图片

        pic_dist=distance_all[i]
        max_index=max_index_all[i]
        max_dist=pic_dist[max_index]

        min_index = max_index
        min_dist = max_dist

        #如果一个图片所有的最大距离都小于0.7
        if max_dist<=0.69:
            index_all.append(max_index)

        else:
            #遍历一个图片的173个距离，找大于0.7的最小距离
            for j in range(len(pic_dist)):
                if pic_dist[j]>0.69 and min_dist>pic_dist[j]:
                    min_index=j
                    min_dist=pic_dist[j]

            index_all.append(min_index)

    return index_all




def changFace(imageName,distance_all):
    """
    根据ImageName distance_all_1来换脸
    :param imageName:
    :param distance_all:  (173,173)
    :return:
    """
    max_index=np.argmax(distance_all,axis=1) #每一行的最大索引
    # min_index=getindex(distance_all)


    for i in range(len(imageName)):
        #遍历每一张图片 找到最大值索引
        img1_path=IMAGES_PATH+imageName[i]
        img1_path=img1_path[0]

        print('pic:{},path={}'.format(imageName[i],img1_path))
        pic_max_index=max_index[i]
        img2_path=IMAGES_PATH+imageName[pic_max_index]
        img2_path=img2_path[0]
        print('max dist is {},path={}'.format(distance_all[i][pic_max_index], img2_path))

        output_path=OUTPUT_PATH+imageName[i]
        output_path=output_path[0]
        print('output path:{}'.format(output_path))

        #换脸部分
        # fc=FaceChanger()
        # fc.load_images(img1_path,img2_path)
        # fc.run(showProcedure=False,saveResult=True,output_path=output_path)

        # 换脸部分
        fc = FaceChangerAll()
        fc.load_images(img1_path, img2_path)
        fc.run(showProcedure=False, saveResult=True, output_path=output_path)




if __name__ == '__main__':

    IMAGES_DEV_NEW = 'H:/cmq/securityAI_round1_dev_new.csv'
    imageName=get_image_names(IMAGES_DEV_NEW)

    print(imageName.shape)
    imageName_1=imageName[0:173,:]
    print(imageName_1.shape)
    print(imageName_1[0])
    print(imageName_1[-1])

    # get_distance_all(imageName=imageName_1)

    distance_all_1=pd.read_csv(current_path+'/data/distance_all_1.csv',header=None)
    distance_all_1_np=np.array(distance_all_1)
    print(distance_all_1_np.shape)
    print(distance_all_1_np[0])

    changFace(imageName_1,distance_all_1_np)

'''    
    print('min dist')
    min_index_all=getindex(distance_all_1_np)
    min_dist_all=[]
    for i in range(len(distance_all_1_np)):
        pic_distance=distance_all_1_np[i]
        pic_index=min_index_all[i]
        min_pic_dist=pic_distance[pic_index]
        min_dist_all.append(min_pic_dist)

    print(len(min_dist_all))
    print(type(min_dist_all))
    print(min_dist_all)

    print('max dist')
    max_dist=np.max(distance_all_1_np,axis=1)
    print(max_dist)

    # print(max_dist==min_dist_all)

    compareres=max_dist==min_dist_all
    print(compareres)
    count=0
    for i in range(len(compareres)):
        if compareres[i]==False:
            count+=1
    print(count)

'''
