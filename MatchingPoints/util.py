# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:07:03 2020

@author: Павел Ермаков
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
# построение эпиполярных линий
#
# @param image1_path       путь к 1-му изображению
# @param image2_path       путь ко 2-му изображению
# @param keypoints1        M x 2 матрица координат ключевых точек на 1-м изображении
# @param keypoints2        M x 2 матрица координат ключевых точек на 2-м изображении
# @param inlier_mask       Массив длины M, состоящий из True - False
#                          True - точка 'хорошая'
#                          False - выброс
def plot_epipolar_inliers(image1_path, image2_path, keypoints1, keypoints2, inlier_mask):

    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    both_images = np.column_stack((image1, image2))

    keypoints1 = np.column_stack((keypoints1, np.ones(len(keypoints1))))
    keypoints2 = np.column_stack((keypoints2, np.ones(len(keypoints2))))

    image1_inliers = keypoints1[inlier_mask, :]
    image2_inliers = keypoints2[inlier_mask, :]

    image1_outliers = keypoints1[~inlier_mask, :]
    image2_outliers = keypoints2[~inlier_mask, :]

    offset = image1.shape[1]

    #--------------------------------------------------------------------------


    plt.title("Соответствующие точки")
    plt.imshow(both_images)
    plt.scatter(keypoints1[:,0], keypoints1[:,1], c='r', s=4)
    plt.scatter(keypoints2[:,0] + offset, keypoints2[:,1], c='b', s=4)

    for i in range(image1_inliers.shape[0]):
        plt.plot(
            (image1_inliers[i,0], image2_inliers[i,0] + offset),
            (image1_inliers[i,1], image2_inliers[i,1]),
            linewidth=1.0, c='g')
    plt.savefig('СоответствующиеТочки.png')
    
    #--------------------------------------------------------------------------
    
    plt.figure()
    plt.title("Точки Выбросы")
    plt.imshow(both_images)
    plt.scatter(keypoints1[:,0], keypoints1[:,1], c='r', s=4)
    plt.scatter(keypoints2[:,0] + offset, keypoints2[:,1], c='b', s=4)

    for i in range(image1_outliers.shape[0]):
        plt.plot(
            (image1_outliers[i,0], image2_outliers[i,0] + offset),
            (image1_outliers[i,1], image2_outliers[i,1]),
            linewidth=1.0, c='y')

    plt.savefig('ТочкиВыбросы.png')