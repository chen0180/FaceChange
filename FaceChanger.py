# coding=utf-8
import cv2
import dlib
import os
import numpy as np
import glob


class TooManyFaces(Exception):
    pass


class NoFace(Exception):
    pass


class FaceChanger(object):

    def __init__(self, which_predictor='68'):
        print('Starting your FaceChanger...')

        self.current_path = os.getcwd()
        print('Current path:{0}'.format(self.current_path))

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
        print('Your predictor is:{0}'.format(predictor_name))


        # some parameters
        self.SCALE_FACTOR = 1
        self.FEATHER_AMOUNT = 11

        self.FACE_POINTS = list(range(17, 68))
        self.MOUTH_POINTS = list(range(48, 61))
        self.RIGHT_BROW_POINTS = list(range(17, 22))
        self.LEFT_BROW_POINTS = list(range(22, 27))
        self.RIGHT_EYE_POINTS = list(range(36, 42))
        self.LEFT_EYE_POINTS = list(range(42, 48))
        self.NOSE_POINTS = list(range(27, 35)) #鼻子
        self.JAW_POINTS = list(range(0, 17)) #下巴

        # Points used to line up the images.
        self.ALIGN_POINTS = (self.LEFT_BROW_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_EYE_POINTS +
                             self.RIGHT_BROW_POINTS + self.NOSE_POINTS + self.MOUTH_POINTS)

        self.OVERLAY_POINTS = [
            self.LEFT_EYE_POINTS + self.RIGHT_EYE_POINTS +self.NOSE_POINTS

        ]

        # + self.LEFT_BROW_POINTS + self.RIGHT_BROW_POINTS
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
            print('left {0}; top {1}; right {2}; bottom {3}'.format(face_rect[0].left(), face_rect[0].top(),
                                                                    face_rect[0].right(), face_rect[0].bottom()))
            # box = face_rect[0]
            # shape = predictor(image, box)
            # return np.matrix([[p.x, p.y] for p in shape.parts()])
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
            print('group:',group)

            self.draw_convex_hull(img, landmarks[group], color=1)

        img = np.array([img, img, img]).transpose((1, 2, 0))

        img = (cv2.GaussianBlur(img, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0) > 0) * 1.0
        img = cv2.GaussianBlur(img, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0)

        return img



if __name__ == '__main__':
    fc=FaceChanger()

    image1_path='E:/securityAI/securityAI_round1_images/images/00001.jpg'
    image2_path='E:/securityAI/securityAI_round1_images/images/00002.jpg'
    output_path='E:/securityAI/securityAI_round1_images/test/00001.jpg'

    fc.load_images(image1_path,image2_path)
    fc.run(showProcedure=True,saveResult=True,output_path=output_path)