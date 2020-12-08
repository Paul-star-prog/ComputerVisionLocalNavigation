# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:04:03 2020

@author: Павел Ермаков
"""
import numpy as np


from util import plot_epipolar_inliers


# генератор случайных чисел
np.random.seed(0)


#-------------------------------------------------------------------------------

# 
#
# @param data                 [M x K] матрица ключевых точек на 2-х изображениях
#                               
# @param inlier_threshold     Пороговое значение м/у 'хорошими' точками и прямой линией,
#                             проводимой RANSAC
#                             Точка считается 'хорошей', если расстояние до прямой,
#                             проводимой RANSAC меньше, чем заданный порог в пкс 
#
# @param confidence_threshold Заданный доверительный порог 
#                             При достижении заданного порога, считаем, что нашли
#                             правильную (точную) фундаментальную матрицу F  
#                        
# @param max_num_trials       Начальное количество итераций алгоритма RANSAC
#
# @return best_F              Лучшее значение фундаментальной матрицы F [3 x 3]
#
# @return inlier_mask         Массив длины M, состоящий из True - False
#                             True - точка 'хорошая'
#                             False - выброс
def ransac(data, inlier_threshold, confidence_threshold, max_num_trials):
    # инициализация максимального числа итераций алгоритма
    max_iter = max_num_trials 
    # текущий номер итерации
    iter_count = 0
    # начальное количество 'хороших' точек
    best_inlier_count = 0
    # начальная матрица 'хороших' точек 
    # True - 'хорошая' точка
    # False - выброс
    # dtype - тип данных
    best_inlier_mask = np.zeros(len(data), dtype=np.bool)
    # начальное состояние фундаментальной матрицы 
    best_F = np.zeros((3, 3))
    # необходимое количество точек для определения фундаментальной матрицы F по алгоритму Хартли
    num_Points = 8

    def make_hartley_vector(p1, p2):
        """
        Функция расчета вектора Хартли
        Входы:
        p1: ключевая точка на 1-м изображении (x,y)
        p2: ключевая точка на 2-м изображении (x,y)
        Выход:
        Массив: вектор Хартли
        """
        u1, v1 = p1
        u2, v2 = p2
        return np.array([u1*u2, u1*v2, u1, v1*u2, v1*v2, v1, u2, v2, 1])

    def make_hartley_matrix(left, right):
        """
        Функция расчета матрицы Хартли (состоящая из векторов Хартли)
        Входы:
        left: массив ключевых точек на 1-м изображении
        right: массив ключевых точек на 2-м изображении
        Выходы:
        A: матрица Хартли [num_Points x 9]
        """
        # количество точек
        m = len(left)
        # инициализация матрицы [m x 9]
        A = np.zeros((m, 9))
        for i in range(m):
            A[i,:] = make_hartley_vector(left[i], right[i])
        return A

    def compute_F(A):
        """
        Функция, возвращающая фундаментальную матрицу F
        Входы:
        A: матрица Хартли [num_Points x 9]
        Выходы:
        F: матрица фундаментальная [3 x 3]
        """
        # SVD разложение прямоугольной матрицы A
        u, s, vT = np.linalg.svd(A)
        # берем последнюю строку в матрице vT, делаем из нее матрицу [3 x 3]
        # order='F' - массив хранится в столбчатом виде в памяти
        return np.reshape(vT[-1], [3, 3], order='F')

    def enforce_rank_2(F):
        """
        Функция, возвращающая точную фундаментальную матрицу F 
        (аппроксимация матрицы F разложением SVD)
        """
        # SVD разложение матрицы F
        u, s, vT = np.linalg.svd(F)
        # последняя строка в матрице s - зануляется
        s[-1] = 0
        return u * np.diag(s) * vT

    def compute_normalization_mat(pts):
        """
        Функция, вычисляющая нормированную матрицу 
        Вход: 
        pts: матрица ключевых точек [num_Points x 2] на одном изображении
        Выход:
        Нормированная матрица [3 x 3]
        """
        # вычисление средних значений координат ключевых точек (pts_X_mean, pts_Y_mean)
        # axis = 0 - суммирование по столбцам
        translate = np.mean(pts, axis=0)
        # вычитание среднего значения
        translated = pts - translate
        # вычисление нормы матрицы translated
        # axis = 1 - значения берутся по строкам
        distances = np.linalg.norm(translated, axis=1)
        # масштабный коэффициент
        scale = np.sqrt(2) / np.mean(distances)
        result = np.array([[scale,     0,     scale * -translate[0]],
                           [    0, scale,     scale * -translate[1]],
                           [    0,     0,                        1]])
        return result

    def normalize_points(pts):
        """
        Функция, нормирующая ключевые точки
        Вход:
        pts: матрица ключевых точек [num_Points x 2] на одном изображении
        Выход:
        res: матрица нормированных ключевых точек [num_Points x 2]
        """
        # Учет среднего значения ключевых точек
        # axis = 0 - суммирование по столбцам
        res = pts - np.mean(pts, axis=0)
        # вычисление среднего расстояния до исходной точки
        # axis = 1 - значения берутся по строкам
        distances = np.linalg.norm(res, axis=1)
        mean_dist = np.mean(distances)
        # масштабный коэффициент
        scale = mean_dist / np.sqrt(2)
        # нормирование ключевых точек
        res /= scale
        return res

    def hartley(points):
        """
        Функция, реализующая 8-и точечный алгоритм Хартли (нормализованный)
        для отыскания фундаментальной матрицы
        Вход: ключевые точки на 2-х изображениях [num_Points x 4]
        Выход: фундаментальная матрица F [3 x 3]
        """
        # вычисление матриц трансформации
        T1 = compute_normalization_mat(points[:,:2])
        T2 = compute_normalization_mat(points[:,2:])

        # нормирование ключевых точек на 2-x изображениях
        left_img_pts = normalize_points(points[:,:2])
        right_img_pts = normalize_points(points[:,2:])

        # расчет фундаментальной матрицы
        # формирование матрицы Хартли
        A = make_hartley_matrix(left_img_pts, right_img_pts)
        # получение из матрицы Хартли --> фундаментальной матрицы
        F = compute_F(A)
        # уточнение фундаментальной матрицы F
        F = enforce_rank_2(F)
        # перевод фундаментальной матрицы на 2-е изображение
        F = np.dot(T2.T, np.dot(F, T1))

        return F


    # Цикл продолжается до тех пор пока не будет достигнуто максимальное количество итераций 
    while iter_count < max_iter:
        # увеличение текущего счетчика итераций
        iter_count += 1

        # случайным образом выбираем из всего количества строчек данных (data) ключевых точек
        # 8 индексов ключевых точек
        # replace - False, индексы строк матрицы не повторяются
        idxs = np.random.choice(len(data), num_Points, replace=False)
        # получение ключевых точек из матрицы data по индексам idxs 
        points = data[idxs]
        
        # расчет матрицы F по 8-ми точечному алгоритму Хартли
        F = hartley(points)
        
        # координаты ключевых точек на первом изображении
        #             | X_keypoint_1  X_keypoint_2   ...   ...     X_keypoint_M |
        # first_img = | Y_keypoint_1  Y_keypoint_2   ...   ...     Y_keypoint_M |
        #             |     1               1        ...   ...           1      |
        first_img = np.column_stack((data[:,:2],np.ones(len(data)))).T
        # рассчитываем эпиполяру: epipolar = F * first_img
        #                   |  X_epipolar_1  X_epipolar_2  ...  ...  X_epipolar_M  |
        # epipolar_lines =  |  Y_epipolar_1  Y_epipolar_2  ...  ...  Y_epipolar_M  |
        #                   |  Z_epipolar_1  Z_epipolar_2  ...  ...  Z_epipolar_M  |
        epipolar_lines = np.dot(F, first_img)

        # Расчет нормы эпиполяры
        # axis = 0, суммирование по столбцам
        norms = np.linalg.norm(epipolar_lines[:2,:], axis=0)
        # Нормировка эпиоляры
        epipolar_lines /= norms

        # Для каждой точки на втором изображении
        # находим расстояние до эпиполяры
        
        # координаты ключевых точек на втором изображении
        #              | X_keypoint_1  X_keypoint_2   ...   ...     X_keypoint_M|
        # second_img = | Y_keypoint_1  Y_keypoint_2   ...   ...     Y_keypoint_M|
        #              |     1               1        ...   ...           1     |        
        second_img = np.column_stack((data[:,2:],np.ones(len(data)))).T
        # размерность массива координат ключевых точек на втором изображении должны совпадать
        # с размерностью массива координат точек эпиполяры
        # при невыполнении данного условия, assert сигнализирует об ошибки
        assert second_img.shape == epipolar_lines.shape
        # расчет расстояния до эпиполяры
        # axis = 0, суммирование по столбцам
        distances = np.sum(second_img * epipolar_lines, axis = 0)
        # полученное расстояние меньше заданного порога 
        # добавляем в массив inlier_mask значение True
        # иначе False
        inlier_mask = (distances < inlier_threshold)
        # Подсчет элементов со значением True
        inlier_count = np.count_nonzero(inlier_mask)

        # если количество 'хороших' точек больше, чем заданное количество, то
        # обновляем решение
      
        if inlier_count > best_inlier_count:
            # лучшее число 'хороших' точек == текущему количеству 'хороших' точек
            best_inlier_count = inlier_count
            # коэффициент 'хороших' точек:
            # количество хороших точек / количетсво поданных исходных точек
            # значение коэффициента от 0 до 1
            best_inlier_ratio = inlier_count / float(len(data))
            best_inlier_mask = inlier_mask
            best_F = F

            # если количество 'хороших' точек совпало с количеством
            # исходных данных, то прерываем цикл
            if inlier_count == len(data):
                break


            # основываясь на коэффициенте 'хороших' точек, пересчитаем макисмальное
            # число итераций, необходимых для достижения заданной вероятности 
            # получения хорошего решения
            
            # вычисление вероятности того, что выбранные точки являются 'хорошими'
            # best_inlier_ratio^ num_Points
            prob_S_samples_are_inliers = np.power(best_inlier_ratio, num_Points)
            
            # есть вероятность того, что количесвто 'хороших' точек мало (->0)
            # для стабильного вычисления знаменателя в формуле для вычисления
            # количества итераций
            # воспользуемся формулой:
            # log(1 - p) = log(exp(0) - exp(log(p)))
            #            = log(exp(log(p)) * (exp(0)/exp(log(p)) - 1))
            #            = log(p) + log(1 / p - 1)
            denom = (np.log(prob_S_samples_are_inliers) + 
                     np.log(1. / prob_S_samples_are_inliers - 1.))
            
            # является ли знаменатель бликим к 0
            if not np.isclose(denom, 0.):
                # вычисляем новое значение максимального числа итераций N:
                # N  = log(1 - p) / denom
                N = np.log(1. - confidence_threshold) / denom
                # выбираем min значение числа итераций из рассчитанного и заданного
                max_iter = min(max_iter, N)


    # финальный расчет матрицы F
    inlier_data = data[best_inlier_mask]
    F = hartley(inlier_data)

    inlier_ratio = best_inlier_count / float(len(data))
    
    return best_F, best_inlier_mask


#-------------------------------------------------------------------------------

# основная функция программы main
def main(args):
    # Считываем ключевые точки 1-го изображения, данные в исходном файле разделены ,
    keypoints1 = np.loadtxt(args.keypoints1, delimiter=",")
    # Считываем ключевые точки 2-го изображения, данные в исходном файле разделены ,
    keypoints2 = np.loadtxt(args.keypoints2, delimiter=",")
    
    # При невыполнении нижеперечисленных условий, assert будет сигнализировать об ошибки
    assert keypoints1.shape[1] == 2, "Массив точек должен иметь размерность M x 2"
    assert keypoints2.shape[1] == 2, "Массив точек должен иметь размерность M x 2"
    assert keypoints1.shape[0] == keypoints2.shape[0],"Строки в 2-x массивах должны быть равны"
    # Объединение 2-х массивов ключевых точек
    # в матрицу M x 4:
    # x_keypoint1 | y_keypoint1|x_keypoint2|y_keypoint2
    data = np.column_stack((keypoints1, keypoints2))

    
    F, inlier_mask = ransac(data, args.inlier_threshold,
                            args.confidence_threshold, args.max_num_trials)

    plot_epipolar_inliers(
      args.image1, args.image2, keypoints1, keypoints2, inlier_mask)
        

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Соответсвие ключевых точек на 2-х изображениях",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #
    # Входные данные
    #
    
    # передача пути файла изображения №1
    parser.add_argument("--image1", type=str, default= r'test_0_0_0.jpg',
        help="first image file")
    # передача пути файла изображения №2
    parser.add_argument("--image2", type=str, default=r'test_20_10_8.jpg',
        help="second image file")
    # передача пути файла ключевых точек на изображении №1
    parser.add_argument(
        "--keypoints1", type=str,
       default=r'test_0_0_0.csv',
        help="keypoints CSV file for the first image, with each line storing "
            "(x,y) pixel coordinates")
    # передача пути файла ключевых точек на изображении №2
    parser.add_argument(
        "--keypoints2", type=str,
        default=r'test_20_10_8.csv',
        help="keypoints CSV file for the second image, with each line storing "
            "(x,y) pixel coordinates")

    #
    # Входные данные для RANSAC
    #
    
    # Пороговое значение м/у 'хорошими' точками и прямой линией,проводимой RANSAC в пикселях
    parser.add_argument("--inlier_threshold", type=float, default=2.,
        help="point-to-line distance threshold, in pixels, to use for RANSAC")
   
    # Остановка алгоритма, когда вероятность того, что была найдена правильная модель достигнет данного порога
    parser.add_argument("--confidence_threshold", type=float, default=0.99,
        help="stop RANSAC when the probability that a correct model has been "
             "found reaches this threshold")
    # Максимальное количество итераций для алгоритма
    parser.add_argument("--max_num_trials", type=float, default=10000,
        help="maximum number of RANSAC iterations to allow")

    args = parser.parse_args()
    # Вызов main
    main(args)