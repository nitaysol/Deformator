import cv2
import numpy as np
import math

drawing = False
secondClick = False


def compress_for_point(point_x, point_y, rad):
    dis = calculate_distance(point_x, point_y, x1, y1) #calculating distance between point to center of second circle
    influence = (abs(dis - rad))/rad
    return influence * vecSecondToFirst[0]+point_x, influence * vecSecondToFirst[1]+point_y


def nearest_neighbor(point_x, point_y):
    rounded_y = int(math.floor(point_y))
    rounded_x = int(math.floor(point_x))
    return img2[rounded_y][rounded_x]


def b_linear(point_x, point_y):
    f_y = int(math.floor(point_y))
    c_y = int(math.ceil(point_y))
    f_x = int(math.floor(point_x))
    c_x = int(math.ceil(point_x))
    if f_x == c_x:
        R1 = img2[f_y][f_x]
        R2 = img[c_y][f_x]
    else:
        R1 = ((c_x - point_x)/(c_x - f_x)) * img2[f_y][f_x] + ((point_x - f_x) / (c_x - f_x)) *img2[f_y][c_x]
        R2 = ((c_x - point_x) / (c_x - f_x)) * img2[c_y][f_x] + ((point_x - f_x) / (c_x - f_x)) * img2[c_y][c_x]
    if c_y == f_y:
        return R1
    return ((c_y - point_y) / (c_y - f_y)) * R1 + ((point_y - f_y) / (c_y - f_y)) * R2


def equations(point_x, point_y, orig_x, orig_y):
    y1 = point_y - 1
    y2 = point_y - 2
    y3 = point_y + 1
    y4 = point_y
    x1 = point_x - 1
    x2 = point_x - 2
    x3 = point_x + 1
    x4 = point_x

    eq1 = np.array([f(x1), f(x2), f(x3), f(x4)])
    result1 = np.linalg.solve(eq1, np.array([img2[y1,x1], img2[y1,x2], img2[y1,x3], img2[y1,x4]]))
    result2 = np.linalg.solve(eq1, np.array([img2[y2,x1], img2[y2,x2], img2[y2,x3], img2[y2,x4]]))
    result3 = np.linalg.solve(eq1, np.array([img2[y3, x1], img2[y3, x2], img2[y3, x3], img2[y3, x4]]))
    result4 = np.linalg.solve(eq1, np.array([img2[y4, x1], img2[y4, x2], img2[y4, x3], img2[y4, x4]]))
    res_x1 = result1[0]*orig_x**3 + result1[1]*orig_x**2 + result1[2]*orig_x + result1[3]
    res_x2 = result2[0]*orig_x**3 + result2[1]*orig_x**2 + result2[2]*orig_x + result2[3]
    res_x3 = result3[0]*orig_x**3 + result3[1]*orig_x**2 + result3[2]*orig_x + result3[3]
    res_x4 = result4[0]*orig_x**3 + result4[1]*orig_x**2 + result4[2]*orig_x + result4[3]

    eq2 = np.array([f(y1), f(y2), f(y3), f(y4)])
    result5 = np.linalg.solve(eq2, np.array([res_x1, res_x2, res_x3, res_x4]))

    return (result5[0]*orig_y**3 + result5[1]*orig_y**2 + result5[2]*orig_y + result5[3])


def f(x):
    return ([x**3, x**2, x, 1])


def cubic(point_x, point_y):
    f_y = int(math.floor(point_y))
    f_x = int(math.floor(point_x))
    return equations(f_x, f_y, point_x, point_y)


def calculate_distance(point1_x, point1_y, point2_x, point2_y):
    return math.sqrt((point2_x - point1_x)**2 + (point2_y - point1_y)**2)


def draw_circle(event, x, y, unused1, unused2):
    global x1, y1, x2, y2, drawing, radius, num, img, img2, imgCubic, imgBlinear, imgNearest, secondClick, first_circle, second_circle, vecSecondToFirst
    first_circle = set()   #Second Circle pixels list
    second_circle = set()  #First Circle pixels list
    far_shape = set()      #Second circle - First Circle
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y
        if not secondClick:
            x2, y2 = x, y
        if secondClick:
            vecSecondToFirst = np.array([x2-x1, y2-y1])
            cv2.circle(img, (x1, y1), radius, (255, 0, 0), 1)
            for b in range(img.shape[0]):
                for a in range(img.shape[1]):
                    if calculate_distance(x2, y2, a, b) < radius:
                        first_circle.add((b, a))
                    elif calculate_distance(x1, y1, a, b) < radius:
                        second_circle.add((b, a))
            far_shape = second_circle.difference(first_circle)
            for (b, a) in far_shape:
                x_res, y_res = compress_for_point(a, b, radius)
                imgBlinear[b][a] = b_linear(x_res, y_res)
                imgNearest[b][a] = nearest_neighbor(x_res, y_res)
                imgCubic[b][a] = cubic(x_res, y_res)
            for (b, a) in first_circle:
                x_res, y_res = compress_for_point(a, b, calculate_distance(x1, y1, x2, y2) + radius)
                imgBlinear[b][a] = b_linear(x_res, y_res)
                imgNearest[b][a] = nearest_neighbor(x_res, y_res)
                imgCubic[b][a] = cubic(x_res, y_res)
        img_all = np.hstack((imgNearest, imgBlinear, imgCubic))
        cv2.imshow(result_window, img_all)
        radius = int(math.hypot(0, 0))

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            a, b = x, y
            if a != x & b != y:
                img = img2.copy()
                radius = int(math.hypot(a - x1, b - y1))
                cv2.circle(img, (x1, y1), radius, (255, 0, 0), 1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        num += 1
        radius = int(math.hypot(x - x1, y - y1))
        if radius > 0 & secondClick == False:
            secondClick = True
            cv2.circle(img, (x1,y1), radius, (255, 0, 0), 1)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(img, '_'.join(['label', str(num)]), (x + 20, y + 20), font, 1, (200, 255, 155), 1, cv2.LINE_AA)
            # img2 = img.copy()


if __name__ == "__main__":
    num = 0
    windowName = 'Drawing'
    result_window = 'Result'
    img = cv2.imread('miz.png')
    img2 = img.copy()
    imgBlinear = img2.copy()
    imgNearest = img2.copy()
    imgCubic = img2.copy()
    cv2.namedWindow(windowName)
    cv2.namedWindow(result_window)
    cv2.setMouseCallback(windowName, draw_circle)
    while True:
        cv2.imshow(windowName, img)
        if cv2.waitKey(20) == 27:
            break

    cv2.destroyAllWindows()
