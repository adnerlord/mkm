from PIL import Image, ImageDraw
import os
import numpy as np
import cv2
from scipy.ndimage import maximum_filter


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# бинаризация изображения
def map_binarization(image_file, save_file):
    image = Image.open(image_file).convert('RGB')
    draw = ImageDraw.Draw(image)

    width, height = image.size

    pix = image.load()
    for i in range(1, width):
        for j in range(1, height):
            a = pix[i, j][0]
            b = pix[i, j][1]
            c = pix[i, j][2]

            if a > 200 and b > 200 and c > 200:
                a = b = c = 255
            else:
                a = b = c = 0

            draw.point((i, j), (a, b, c))

    image.save(save_file)
    del draw


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# убираем фон и географические данные
def map_filter(image_file, save_file):
    image = Image.open(image_file).convert('RGB')
    draw = ImageDraw.Draw(image)

    width, height = image.size

    pix = image.load()
    for i in range(1, width):
        for j in range(1, height):
            a = 255 - pix[i, j][0]
            b = 255 - pix[i, j][1]
            c = 255 - pix[i, j][2]

            a = 0 if a == 32 else a

            if a > 200 and b > 200 and c > 200:
                a = b = c = 255
            else:
                a = b = c = 0

            draw.point((i, j), (a, b, c))

    for i in range(0, width):
        draw.point((i, 0), (0, 0, 0))
        draw.point((i, height - 1), (0, 0, 0))
        draw.point((i, height - 2), (0, 0, 0))

    for j in range(0, height):
        draw.point((0, j), (0, 0, 0))
        draw.point((width - 1, j), (0, 0, 0))
        draw.point((width - 2, j), (0, 0, 0))

    image.save(save_file)
    del draw


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# рисуем рамки найденных совпадений
def draw_frame(img, coord, thickness=-1, color=(0, 0, 0)):
    ind = 0 if thickness != -1 else int(thickness / 2)

    res = img.copy()
    for c in coord:
        top_left = (c[0] - ind, c[1] - ind)
        bottom_right = (c[0] + c[2] + ind, c[1] + c[3] + ind)
        cv2.rectangle(res, top_left, bottom_right, color=color, thickness=thickness)
    return res


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def template_matcher(image_file, accuracy=0.0):
    # найти совпадения по шаблонам
    def find_templ(img, img_tpl):
        # размер шаблона
        h, w = img_tpl.shape

        # строим карту совпадений с шаблоном
        match_map = cv2.matchTemplate(img, img_tpl, cv2.TM_CCOEFF_NORMED)

        max_match_map = np.max(match_map)  # значение карты для области максимально близкой к шаблону
        print(max_match_map)
        if (max_match_map < 0.71):  # совпадения не обнаружены
            return []

        # коэффициент "похожести", 0 - все, 1 - точное совпадение
        if accuracy == 0.0:
            if h < 100 and w < 100:
                a = 0.8
            elif (h > 20 or h < 21) and (w > 14 or w < 16):
                a = 0.2
            else:
                a = 0.7
        else:
            a = accuracy

        # отрезаем карту по порогу
        match_map = (match_map >= max_match_map * a) * match_map

        # выделяем на карте локальные максимумы
        match_map_max = maximum_filter(match_map, size=min(w, h))
        # т.е. области наиболее близкие к шаблону
        match_map = np.where((match_map == match_map_max), match_map, 0)

        # координаты локальных максимумов
        ii = np.nonzero(match_map)
        rr = tuple(zip(*ii))

        res = [[c[1], c[0], w, h] for c in rr]

        return res

    f = image_file
    bu = "../data/templ/"

    templ = [os.path.join(bu, b) for b in os.listdir(bu) if os.path.isfile(os.path.join(bu, b))]

    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)

    tr_coord = []
    high_coord = []
    low_coord = []

    for t in templ:
        print(t)

        img_tpl = cv2.imread(t, cv2.IMREAD_GRAYSCALE)

        if t[-5:-4] == "H":
            low_coord.append(find_templ(img, img_tpl))
        elif t[-5:-4] == "B":
            high_coord.append(find_templ(img, img_tpl))
        else:
            tr_coord.append(find_templ(img, img_tpl))

    return tr_coord, high_coord, low_coord


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def draw_frames_pack(coords, image_file, save_file, thickness=-1, color=(0, 0, 0)):
    img = cv2.imread(image_file, cv2.IMREAD_COLOR)

    for c in coords:
        img = draw_frame(img, c, thickness, color)
    # for c in high_coord:
    #     img = draw_frames(img, c, 2, (20, 20, 255))
    # for c in low_coord:
    #     img = draw_frames(img, c, 2, (255, 30, 30))
    cv2.imwrite(save_file, img)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# def sift_key_point_matcher():
#     # поворачиваем и масштабируем основную картинку
#     def load_img_big(f):
#         coef = 1
#         img1 = cv2.imread(f, cv2.IMREAD_GRAYSCALE)  # queryImage
#         dsize = (int(img1.shape[1] * coef), int(img1.shape[0] * coef))
#         img1 = cv2.resize(src=img1, dsize=dsize, interpolation=cv2.INTER_CUBIC)
#         # rows, cols = img1.shape
#         # M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 14, 1)
#         # img1 = cv2.warpAffine(img1, M, (cols, rows))
#
#         return img1
#
#     MIN_MATCH_COUNT = 10  # порог минимального количества совпадений ключевых точек
#     DIST_COEFF = 0.85
#
#     f = "../data/result/res.bmp"
#     bu = "../data/templ2/"
#
#     print("[i] считываем картинку")
#     img_big = load_img_big(f)
#
#     sift = cv2.xfeatures2d.SIFT_create()  # Initiate SIFT detector
#
#     matcher = cv2.BFMatcher()  # BFMatcher with default params
#
#     # find the keypoints and descriptors with SIFT
#     kp_big, des_big = sift.detectAndCompute(img_big, None)
#
#     templ = [os.path.join(bu, b) for b in os.listdir(bu) if os.path.isfile(os.path.join(bu, b))]
#
#     for t in templ:
#         print(t)
#         img_tpl = cv2.imread(t, cv2.IMREAD_GRAYSCALE)
#
#         print("[i] ищем особые точки")
#         kp_tpl, des_tpl = sift.detectAndCompute(img_tpl, None)
#
#         print("[i] сопоставляем особые точки")
#         matches = matcher.knnMatch(des_tpl, des_big, k=2)
#
#         good = []
#         for m, n in matches:
#             if m.distance < n.distance * DIST_COEFF:
#                 good.append(m)
#
#         if len(good) < MIN_MATCH_COUNT:
#             print("[w] количество совпадений недостаточно - %d/%d" % (len(good), MIN_MATCH_COUNT))
#         else:
#             print("[i] найдено %d совпадений" % len(good))
#
#             src_pts = np.float32([kp_tpl[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#             dst_pts = np.float32([kp_big[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#             M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  # матрица преобразования координат точек
#
#             # координаты точек рамки шаблона
#             h, w = img_tpl.shape  # размеры шаблона
#             pts = np.asarray([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]], dtype=np.float32).reshape(-1, 1, 2)
#
#             # выполняем преобразования координат точек рамки шаблона
#             dst = cv2.perspectiveTransform(pts, M)
#
#             dst = [np.int32(np.abs(dst))]  # обрезаем рамку вылезшую за пределы картинки
#
#             # рисуем рамку вокруг найденного объекта
#             img_res = cv2.cvtColor(img_big, cv2.COLOR_GRAY2BGR)
#             img_res = cv2.polylines(img_res, dst, True, (0, 255, 255), 2, cv2.LINE_AA)
#
#             # рисуем совпадения контрольных точек
#             matchesMask = mask.ravel().tolist()
#             draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
#                                singlePointColor=None,
#                                matchesMask=matchesMask,  # draw only inliers
#                                flags=2)
#             img_res = cv2.drawMatches(img_tpl, kp_tpl, img_res, kp_big, good, None, **draw_params)
#
#             # записываем результат
#             tn = os.path.splitext(os.path.basename(t))[0]
#             cv2.imwrite("../data/result/res-123.bmp" % tn, img_res)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def erode(image_file, save_file):
    morf_kernel = np.ones((5, 5))

    f = image_file

    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)

    img_ero = cv2.erode(img, morf_kernel, iterations=1)

    cv2.imwrite(save_file, img_ero)


def dilate(image_file, save_file, iterations=1):
    morf_kernel = np.ones((5, 5))

    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    img_dil = cv2.dilate(img, morf_kernel, iterations=iterations)

    cv2.imwrite(save_file, img_dil)


def morph(image_file, save_file):
    img = cv2.imread(image_file)
    img_bw = 255 * (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 5).astype('uint8')

    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (29, 29))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

    mask = np.dstack([mask, mask, mask]) / 255
    out = img * mask

    # cv2.imshow('Output', out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(save_file, out)


def main():
    map_file_1 = "../data/maps/12бз-08012016.GIF"
    save_file_1 = "../data/result/1_1.bmp"
    save_file_2 = "../data/result/1_2.bmp"
    save_file_3 = "../data/result/1_3.bmp"
    save_file_4 = "../data/result/1_4.bmp"
    save_file_5 = "../data/result/1_5.bmp"
    save_file_6 = "../data/result/1_6.bmp"
    save_file_7 = "../data/result/1_777.bmp"

    # инверсия, удаление фона
    map_filter(map_file_1, save_file_1)

    # получение координат по найденным шаблонам
    coord_trash, coord_high, coord_low = template_matcher(save_file_1)

    # стирание ненужных данных
    draw_frames_pack(coord_trash, save_file_1, save_file_2)

    # выделение обозначений H/В давления
    draw_frames_pack(coord_high, save_file_2, save_file_3)
    draw_frames_pack(coord_low, save_file_3, save_file_3)

    # map_binarization(save_file_2, save_file_3)
    erode(save_file_3, save_file_4)
    dilate(save_file_4, save_file_5)

    # выделение обозначений H/В давления
    draw_frames_pack(coord_high, save_file_2, save_file_3, color=(0, 0, 255))
    draw_frames_pack(coord_low, save_file_3, save_file_3, color=(255, 0, 0))

    _, coord_high_sensitive, coord_low_sensitive = template_matcher(save_file_3, 0.7)
    draw_frames_pack(coord_high_sensitive, save_file_3, save_file_6, color=(0, 0, 255))
    draw_frames_pack(coord_low_sensitive, save_file_6, save_file_6, color=(255, 0, 0))

    # Улучшенный морфологический фильтр
    morph(save_file_6, save_file_7)


if __name__ == "__main__":
    main()
