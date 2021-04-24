#  -*- encode: utf-8 -*-
import cv2 as cv
import numpy as np


"""图片的读取和保存"""
image2 = cv.imread('e:/cv33/lena.png')  # 读取图片
cv.namedWindow('haha', cv.WINDOW_AUTOSIZE)  # 创建窗口
cv.imshow('haha', image2)  # 显示图片
cv.waitKey(0)  # 延迟等待毫秒
cv.destroyAllWindows()  # 毁灭窗口
image2_gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)  # 图片灰度化
cv.imwrite('e:/ss.png', image2_gray)  # 保存图片


"""视频和摄像头数据的读取和操作"""
def video_demo():
    capture = cv.VideoCapture('e:/cv33/you.mp4')  # 连接视频地址或摄备，摄像头编号从0开始
    ss = 1
    while True:
        ss += 1
        path = 'e:/cv33/tr/%s.png' % ss
        ret, frame = capture.read()  # 第1个反回True(有值)或False(无值)，第2个反回每一帧图像
        frame = cv.flip(frame, 1)  # 对每一帧反转
        cv.imwrite(path, frame)
        cv.imshow('video', frame)  # 将每一帧显示出来
        c = cv.waitKey(50)  # 响应用户操作50豪秒，什么时候停掉
        if c == 27:  # 这是啥意思？
            break
video_demo()
cv.waitKey(0)
cv.destroyAllWindows()


"""Numpy操作"""
t1 = cv.getTickCount()  # 获取CPU多少针
# 执行过程
t2 = cv.getTickCount()  # 获取CPU多少针
print((t2 - t1)/cv.getTickFrequency())  # 除以每秒针数，等于执行秒数

cv.namedWindow('hehe', cv.WINDOW_AUTOSIZE)
def create_image():  # 生成一张多通道蓝色的图片，通道顺序B,G,R。单通道图片就是[400, 400, 1]
    img = np.zeros([400, 400, 3], np.uint8)
    img[:, :, 0] = np.ones([400, 400]) * 255
    cv.imshow('hehe', img)
create_image()
cv.waitKey(0)  # 延迟等待毫秒


"""色彩空间：有RGB（方块）,HSV（圆柱）,HIS（圆锥）,YCrCb,YUV"""
cv.bitwise_not(image2)  # 对像素取反
# RGB色彩空间，用于图像读取和保存
# HSV色彩空间，用于找到图像中颜色显著的物体，H是0-180，其他是0-255
# HIS色彩空间，I是灰度，S是饱和度
# YCrCb色彩空间，提取人的皮肤颜色
# YUV色彩空间，Linux上的专用色彩空间
gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)  # 色彩空间转换，下同
hsv = cv.cvtColor(image2, cv.COLOR_BGR2HSV)
yuv = cv.cvtColor(image2, cv.COLOR_BGR2YUV)
ycrcb = cv.cvtColor(image2, cv.COLOR_BGR2YCrCb)

def extrace_object():  # 颜色过滤
    capture = cv.VideoCapture('e:/cv33/youo.mp4')
    while True:
        ret, frame = capture.read()
        if ret == False:
            break
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # 色彩跟踪转换
        lower_hsv = np.array([156, 43, 46])  # 跟踪红色的最小值
        upper_hsv = np.array([180, 255, 255])  # 跟踪红色的最大值
        mask = cv.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)  # 得到二值图像
        dst = cv.bitwise_and(frame, frame, mask=mask)
        cv.imshow('video', frame)
        cv.imshow('mask', dst)
        c = cv.waitKey(40)
        if c == 27:  # 27是escape的意思？
            break
extrace_object()
cv.waitKey(0)
cv.destroyAllWindows()


image2 = cv.imread('e:/cv33/lena.png')  # 读取图片

b, g, r = cv.split(image2)  # 通道分离
cv.imshow('blue', b)
cv.imshow('green', g)
cv.imshow('red', r)
cv.waitKey(0)
cv.destroyAllWindows()

image2[:, :, 2] = 0  # 对图片第3个通道赋值0
cv.imshow('change 2', image2)
cv.waitKey(0)
cv.destroyAllWindows()

image2 = cv.merge([b, g, r])  # 通道合并
cv.imshow('change 2', image2)
cv.waitKey(0)
cv.destroyAllWindows()


"""像素运算（加减乘除用于亮度对比度调节）（与或非用于遮罩层控制）（图像混合，算法运算，几何运算）"""
src1 = cv.imread('e:/cv33/my_mask.png')
src2 = cv.imread('e:/cv33/lena.png')
src1 = src1[0:src2.shape[0], 0:src2.shape[1], 0:src2.shape[2]]  # 按照src2的大小，裁剪src1
src2 = src2[0:src1.shape[0], 0:src1.shape[1], 0:src1.shape[2]]  # 按照src2的大小，裁剪src1
add2 = cv.add(src1, src2)  # 像素相加
sub2 = cv.subtract(src1, src2)  # 像素相减
div2 = cv.divide(src1, src2)  # 像素相除
mul2 = cv.multiply(src1, src2)  # 像素相乘
cv.imshow('mul2', mul2)
cv.imshow('div2', div2)
cv.imshow('add2', add2)
cv.imshow('sub2', sub2)
cv.imshow('src1', src1)
cv.imshow('src2', src2)
cv.waitKey(0)

avg1 = cv.mean(src2)  # 计算图像均值，是计算每个通道
avg2, std = cv.meanStdDev(src2)  # 计算图像均值和标准差，是计算每个通道

bt_and = cv.bitwise_and(src1, src2)  # 与运算
bt_or = cv.bitwise_or(src1, src2)  # 或运算
bt_not = cv.bitwise_not(src2)  # 非运算
cv.bitwise_xor()  # 异或运算，没做例子
cv.imshow('bt_and', bt_and)
cv.imshow('bt_or', bt_or)
cv.imshow('bt_not', bt_not)
cv.waitKey(0)

mask = cv.inRange(hsv, lowerb=1, upperb=2)  # 制作掩模
dst = cv.bitwise_and(src1, src1, mask=mask)  # 提取感兴趣区域

def contrast_brightness_demo(image, c, b):  # 调整图像对比度c，亮度b
    h, w, ch = image.shape
    blank = np.zeros([h, w, ch], image.dtype)
    dst = cv.addWeighted(image, c, blank, 1-c, b)
    cv.imshow('shasha', dst)
image2 = cv.imread('e:/cv33/lena.png')  # 读取图片
contrast_brightness_demo(image2, 1.5, 10)
cv.waitKey(0)

###########################################################################################
"""ROI区域常用来填充图像与泛洪填充"""
image2 = cv.imread('e:/cv33/lena.png')  # 读取图片
face = image2[200:400, 150:350]  # 指定ROI区域
gray_face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)  # 转为灰度图片
original_face = cv.cvtColor(gray_face, cv.COLOR_GRAY2BGR)  # 再转回来，可为啥还是灰的？
image2[200:400, 150:350] = original_face  # 替换ROI区域
cv.imshow('ff', image2)
cv.waitKey(0)
cv.destroyAllWindows()


def fill_color_demo(image):  # 泛洪填充，RGB
    copyImg = image.copy()
    h, w, c = image.shape
    mask = np.zeros([h+2, w+2], np.uint8)
    cv.floodFill(copyImg, mask, (30, 30), (255, 0, 255), (100, 100, 100), (50, 50, 50), cv.FLOODFILL_FIXED_RANGE)
    # (30, 30)起始点，在这里取色。(100, 100, 100)在起始点色彩上每个通道减去这个值。 (50, 50, 50)在起始点上加这个值。
    cv.imshow('fill color_demo', copyImg)


def fill_binary():  # 泛洪填充，二值
    image0 = np.zeros([400, 400, 3], np.uint8)
    image0[20:350, 20:350, :] = 255
    mask = np.ones([image0.shape[0]+2, image0.shape[1]+2, 1], np.uint8)  # 设为1是因为1的地方是不会填充的
    mask[101:301, 101:301] = 0  # 需要填充的地方设置为0，1的地方不会填充，没有变化。
    cv.floodFill(image0, mask, (200, 200), (255, 255, 0), cv.FLOODFILL_MASK_ONLY)
    # (30, 30)起始点，在这里取色。
    cv.imshow('fill color_demo', image0)


###########################################################################################
"""模糊操作（内行看这个）均值模糊，中值模糊，自定义模糊，高斯模糊"""
def blue_demo(image):
    dst = cv.blur(image, (5, 3))  # 均值模糊，可以去噪声
    cv.imshow('blur', dst)

def median_blue_demo(image):
    dst = cv.medianBlur(image, 5)  # 均值模糊，去除椒盐噪声效果非常好
    cv.imshow('blur', dst)

def custom_blue_demo(image):
    # kernal = np.ones([5, 5], np.float32)/25  # 这种是自定义卷积核
    kernal = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 这种自定义的是锐化算子，图片有细节，更立体
    dst = cv.filter2D(image, -1, kernel=kernal)
    cv.imshow('blur', dst)


def clamp(pv):
    if pv > 255:
        return 255
    if pv < 0:
        return 0
    return pv
def gaussian_noise(image):  # 给图片增加高斯噪声
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0, 20, 3)
            b = image[row, col, 0]  # blue
            g = image[row, col, 1]  # green
            r = image[row, col, 2]  # red
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    cv.imshow('gaussian', image)


image2 = cv.imread('e:/cv33/lena.png')
image2 = cv.GaussianBlur(image2, (5, 5), 15)  # (5, 5)和15都是西格玛参数，优先前面的。高斯模糊尤其仰制高斯噪声效果好
cv.imshow('gaussian', image2)
cv.waitKey(0)
cv.destroyAllWindows()


###########################################################################################
###########################################################################################
"""边缘保留滤波（EPF），美颜的基础，高斯双边和均值迁移是实现EPF的两种常用方法"""
def bi_demo(image):  # 高斯双边模糊
    dst = cv.bilateralFilter(image, d=0, sigmaColor=50, sigmaSpace=15)  # d用来推导sigma（在sigma未指定时），d=0忽略之
    cv.imshow('bi', dst)


def shift_demo(image):  # 均值迁移模糊
    dst = cv.pyrMeanShiftFiltering(image, 10, 50)
    cv.imshow('bi', dst)


"""图像直方图"""
def plot_demo(image):  # 统计像素取值分布
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()
def image_hist(image):  # 图像直方图绘制
    color = ('blue', 'green', 'red')
    for i, color in enumerate(color):
        hist = cv.calcHist([image], [i], None, histSize=[256], ranges=[0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()


"""直方图均衡化， 自动提升图像对比度"""
def equalHist_demo(image):  # 全局直方图均衡化
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 均衡化只能对黑白图片，所以要转换为灰度图
    dst = cv.equalizeHist(gray)
    cv.imshow('asa', dst)
def clahe_demo(image):  # 局部直方图均衡化,自适应的
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 均衡化只能对黑白图片，所以要转换为灰度图
    clahe = cv.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
    dst = clahe.apply(gray)
    cv.imshow('asa', dst)


"""直方图比较，比较两个直方图是否相似"""
def create_rgb_hist(image):
    h, w, c = image.shape
    rgbHist = np.zeros([16 * 16 * 16, 1], np.float32)
    bsize = 256 / 16
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            index = np.int(b/bsize) * 16 * 16 + np.int(g/bsize) * 16 + np.int(r/bsize)
            rgbHist[np.int(index), 0] = rgbHist[np.int(index), 0] + 1
    return rgbHist
def hist_compare(image1, image2):
    hist1 = create_rgb_hist(image1)
    hist2 = create_rgb_hist(image2)
    match1 = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)  # 这里的hist一定要float32格式数据
    match2 = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
    match3 = cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR)
    print(match1, match2, match3)


###########################################################################################
"""直方图反向投影"""
def hist2d_demo(image):  # 绘制直方图
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv.imshow('aa', hist)
def back_projection_demo():  # 直方图反向投影
    sample = cv.imread('e:/cv33/football1a.png')
    target = cv.imread('e:/cv33/football1.png')
    roi_hsv = cv.cvtColor(sample, cv.COLOR_BGR2HSV)
    target_hsv = cv.cvtColor(target, cv.COLOR_BGR2HSV)
    cv.imshow('hhh', sample)
    cv.imshow('kkk', target)
    roiHist = cv.calcHist([roi_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])  # [180,256]改成[36,48],分粗些减少碎片
    cv.normalize(roiHist, roiHist, 0, 255, cv.NORM_MINMAX)
    dst = cv.calcBackProject([target_hsv], [0, 1], roiHist, [0, 180, 0, 256], 1)
    cv.imshow('projection', dst)


"""模板匹配"""
def template_demo():  # 只能精确匹配
    tpl = cv.imread('e:/cv33/football1a.png')
    target = cv.imread('e:/cv33/football1.png')
    method = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]
    th, tw = tpl.shape[:2]
    for md in method:
        target_plt = target.copy()
        print(md)
        result = cv.matchTemplate(target, tpl, md)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if md == cv.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0] + tw, tl[1] + th)
        cv.rectangle(target_plt, tl, br, (0, 0, 255), 2)
        cv.imshow('match'+np.str(md), target_plt)
        cv.imshow('find'+np.str(md), result)


###########################################################################################
###########################################################################################
"""图像二值化，有全局二值化和局部二值化两种"""
def threshold_demo(image):  # 全局阈值分割
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 第2个值是指定的，在后面没有参数才会用
    print('threshold %s' % ret)
    cv.imshow('bi', binary)
def local_threshold_demo(image):  # 局部阈值分割
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)  # blockSize必须奇数
    cv.imshow('bi', binary)
def custom_threshold_demo(image):  # 局部阈值分割
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    m = np.reshape(gray, [1, w * h])
    mean = m.sum() / (w * h)
    ret, binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)  # blockSize必须奇数
    cv.imshow('bi', binary)


"""超大图像二值化"""
def big_image_binary(image):
    cw, ch = 256, 256
    h, w = image.shape[:2]
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    for row in range(0, h, ch):
        for col in range(0, w, cw):
            roi = gray[row:row+ch, col:col+cw]
            # ret, dst = cv.threshold(roi, 0 ,255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            dst = cv.adaptiveThreshold(roi, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 127, 20)
            gray[row:row + ch, col:col + cw] = dst
            print(np.std(dst), np.mean(dst))
    cv.imwrite('e:/cv33/big_bi2.png', gray)


###########################################################################################
###########################################################################################
"""图像金字塔"""
def pyramid_demo(image):  # 高斯金字塔降采样
    level = 3
    temp = image.copy()
    pyramid_images = []
    for i in range(level):
        dst = cv.pyrDown(temp)  # 高斯降采样，要求图像的长宽均为2**n+1
        pyramid_images.append(dst)
        cv.imshow('haha'+str(i), dst)
        temp = dst.copy()
    return pyramid_images
def lapalian_demo(image):  # 拉普拉斯金字塔
    pyramid_images = pyramid_demo(image)
    level = len(pyramid_images)
    for i in range(level-1, -1, -1):
        if (i-1) < 0:
            expand = cv.pyrUp(pyramid_images[i], dstsize=image.shape[:2])  # 高斯升采样，要求图像的长宽均为2**n+1
            lpls = cv.subtract(image, expand)  # 拉普拉斯金字塔
            cv.imshow('a'+str(i), lpls)
        else:
            expand = cv.pyrUp(pyramid_images[i], dstsize=pyramid_images[i - 1].shape[:2])  # 高斯升采样
            lpls = cv.subtract(pyramid_images[i - 1], expand)  # 拉普拉斯金字塔
            cv.imshow('a'+str(i), lpls)


###########################################################################################
###########################################################################################
"""图像梯度，sobel算子，拉普拉斯算子"""
def sobel_demo(image):  # sobel算子
    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow('x', gradx)
    cv.imshow('y', grady)
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    cv.imshow('xy', gradxy)
def Scharr_demo(image):  # Scharr算子，是sobel算子的增强版，提取更多边缘，受噪声影响大。
    grad_x = cv.Scharr(image, cv.CV_32F, 1, 0)
    grad_y = cv.Scharr(image, cv.CV_32F, 0, 1)
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow('x', gradx)
    cv.imshow('y', grady)
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    cv.imshow('xy', gradxy)
def laplacian_demo(image):  # 拉普拉斯算子
    dst = cv.Laplacian(image, cv.CV_32F)
    lpls = cv.convertScaleAbs(dst)
    cv.imshow('lpls', lpls)
def custom_demo(image):  # 自定义算子
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    dst = cv.filter2D(image, cv.CV_32F, kernel=kernel)
    cust = cv.convertScaleAbs(dst)
    cv.imshow('cust', cust)


###########################################################################################
###########################################################################################
"""边缘提取，Canny算法很实用也很常用，高斯模糊去噪，灰度转换，梯度计算，图像角度非最大信号抑制，高低梯度双阈值过滤"""
def edge_demo(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)  # canny算法对噪声敏感，所以第一步要降噪，但是不能参数太高
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    edge_output = cv.Canny(xgrad, ygrad, 15, 45)  # 高阈值是低阈值的2到3倍
    edge_output_gray = cv.Canny(gray, 50, 150)  # 高阈值是低阈值的2到3倍，直接对灰度图使用边缘提取
    cv.imshow('canny', edge_output)
    cv.imshow('canny2', edge_output_gray)
    dst = cv.bitwise_and(image, image, mask=edge_output)
    cv.imshow('color', dst)


"""霍夫直线检测，Hough Line Transform。霍夫直线变换要在边缘检测之后"""
def line_detecton(image):  # 先转灰度，再提取边缘，再霍夫。这个得到的是直线，两边限延。
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)  # 3是窗口大小，默认3
    lines = cv.HoughLines(edges, 1, np.pi/180, 200)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0+1000*(-b))  # 乘以1000的原因，要看源码了，暂时不知道。
        y1 = int(y0+1000*(a))
        x2 = int(x0-1000*(-b))
        y2 = int(y0-1000*(a))
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow('asdf', image)
def line_detecton_possible_demo(image):  # 得到线段
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)  # 3是窗口大小，默认3
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow('qwer', image)


"""霍夫圆检测"""
def detect_circles_demo(image):
    dst = cv.pyrMeanShiftFiltering(image, 10, 100)
    cimage = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(cimage, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)
        cv.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)
    cv.imshow('cc', image)


###########################################################################################
###########################################################################################
"""轮廓发现，基于边缘提取寻找轮廓，边缘提取阈值影响大。基于拓补结构扫描"""
def edge_demo_1(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)  # canny算法对噪声敏感，所以第一步要降噪，但是不能参数太高
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    edge_output = cv.Canny(xgrad, ygrad, 30, 90)  # 高阈值是低阈值的2到3倍
    cv.imshow('canny', edge_output)
    return edge_output
def contours_demo(image):
    """dst = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow('bbb', binary)"""
    # 上面是一种方式获取二值图像，下面是另一种方式获取二值图像
    """边缘提取，Canny算法很实用也很常用，高斯模糊去噪，灰度转换，梯度计算，图像角度非最大信号抑制，高低梯度双阈值过滤"""
    binary = edge_demo_1(image)
    cloneImage, contours, heriachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        cv.drawContours(image, contours, i, (0, 0, 255), 1)
        print(i)
    cv.imshow('ct', image)


"""多边形拟合"""
def measure_object(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    print(ret)
    cv.imshow('bi', binary)
    dst = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    outImage, contours, hireachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)
        x, y, w, h = cv.boundingRect(contour)  # 外接矩形四个角的坐标
        rate = min(w, h)/max(w, h)  # 外接矩形的宽高比
        mm = cv.moments(contour)
        cx = mm['m10']/mm['m00']
        cy = mm['m01']/mm['m00']
        cv.circle(dst, (np.int(cx), np.int(cy)), 2, (0, 255, 255), -1)  # 把点画到二值图像上
        # cv.rectangle(dst, (x, y), (x+w, y+h), (0, 0, 255), 2)  # 把框画到二值图像上
        print('area %s' % area)  # 打印目标面积
        approxCurve = cv.approxPolyDP(contour, 4, True)  # 多边形逼近
        if approxCurve.shape[0] > 5:
            cv.drawContours(dst, contours, i, (255, 255, 0), 2)
        if approxCurve.shape[0] == 4:
            cv.drawContours(dst, contours, i, (255, 0, 0), 2)
        if approxCurve.shape[0] == 3:
            cv.drawContours(dst, contours, i, (0, 255, 0), 2)
    cv.imshow('mes', dst)


###########################################################################################
###########################################################################################
###########################################################################################
"""图像形态学（处理灰度与二值图），有两个最基本操作：膨胀与腐蚀"""
"""膨用作用：对象大小增加1个像素（3*3），平滑对象边缘，减少或者填充对象之间的距离"""
"""腐蚀作用：对象大小减少1个像素（3*3），反平滑对象边缘，弱化或者分割图像之间的半岛型连接"""
def erode_demo(image):  # 腐蚀
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)  # cv.THRESH_BINARY，可不取反试试
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = binary.copy()
    for i in range(5):
        dst = cv.erode(dst, kernel)
        cv.imshow('erode'+str(i), dst)
    cv.imshow('image', image)
def dilate_demo(image):  # 膨胀
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)  # cv.THRESH_BINARY，可不取反试试
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))  # 膨胀后形状会趋于kernel形状
    dst = binary.copy()
    for i in range(5):
        dst = cv.dilate(dst, kernel)
        cv.imshow('erode'+str(i), dst)
    cv.imshow('image', image)
def color_demo(image):  # 彩图膨胀会让亮的地方更亮，彩图腐蚀会让黑的地方更黑
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))  # 膨胀后形状会趋于kernel形状
    dst = image.copy()
    for i in range(5):
        dst = cv.erode(dst, kernel)
        cv.imshow('erode'+str(i), dst)
    cv.imshow('image', image)


"""图像形态学之开闭操作（基于膨胀与腐蚀的组合），用在二值图像或灰度图的OCR识别，轮廓匹配，特征分析等"""
"""开操作：先腐食后膨胀，可消除小的噪点"""
"""闭操作：先膨胀后腐食，可以填充小的空洞"""
"""提取水平线（图中横线多）或垂直线（图中竖线多）"""
def open_demo(image):  # 开操作可以保留其他元素，只去掉小噪点，这是与腐蚀不同的地方
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # cv.THRESH_BINARY，可不取反试试
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    cv.imshow('original', image)
    cv.imshow('dst', dst)
def close_demo(image):  # 闭操作
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # cv.THRESH_BINARY，可不取反试试
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    cv.imshow('original', image)
    cv.imshow('dst', dst)
def open_xline_demo(image):  # 开操作提取横的线
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)  # cv.THRESH_BINARY，可不取反试试
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 1))  # （15，1）会把垂直线给腐蚀掉
    dst = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    cv.imshow('original', image)
    cv.imshow('dst', dst)
def open_yline_demo(image):  # 开操作提取竖的线
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)  # cv.THRESH_BINARY，可不取反试试
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 15))  # （1，15）会把水平线给腐蚀掉
    dst = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    cv.imshow('original', image)
    cv.imshow('dst', dst)
def open_line_demo(image):  # 开操作去除干扰线和点
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)  # cv.THRESH_BINARY，可不取反试试
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))  # （3,3）这个核可以删除小的干扰块和细的干扰线
    dst = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)  # cv.MORPH_ELLIPSE结构可以保留圆的区域，删除不规则点
    cv.imshow('original', image)
    cv.imshow('dst', dst)


"""其他形态学操作：顶帽、黑帽、形态学梯度"""
"""顶帽：原图与开操作之间的差值图像"""
"""黑帽：原图与闭操作之间的差值图像"""
"""形态学梯度：基本梯度（用膨胀后的图减去腐蚀后的图），内部梯度（原图减腐蚀后图像），外部梯度（膨胀后图减原图）"""
def top_hat_demo(image):  # 顶帽
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)
    dst = cv.add(dst, 100)
    cv.imshow('tophat', dst)
    cv.imshow('image', image)
def black_hat_demo(image):  # 黑帽
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)
    dst = cv.add(dst, 100)
    cv.imshow('tophat', dst)
    cv.imshow('image', image)
def gradient_demo(image):  # 基本梯度
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.morphologyEx(gray, cv.MORPH_GRADIENT, kernel)
    dst = cv.add(dst, 100)
    cv.imshow('tophat', dst)
    cv.imshow('image', image)
def in_out_gradient_demo(image):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dm = cv.dilate(image, kernel)
    em = cv.erode(image, kernel)
    dst1 = cv.subtract(image, em)  # internal gradient
    dst2 = cv.subtract(dm, image)  # external gradient
    cv.imshow('kk', dst1)
    cv.imshow('jj', dst2)
    cv.imshow('image', image)


###########################################################################################
###########################################################################################
"""分水岭算法，"""
def watershed_demo(src):
    blurred = cv.pyrMeanShiftFiltering(src, 10, 100)  # 图像中间去噪
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow('bi', binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)  # 形态学开操作，去中间小洞
    sure_bg = cv.dilate(mb, kernel, iterations=3)  # 形态学膨胀操作
    cv.imshow('mor', sure_bg)
    dist = cv.distanceTransform(mb, cv.DIST_L2, 3)  # 距离计算
    dist_output = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)  # 标准化
    cv.imshow('dist', dist_output*50)
    ret, surface = cv.threshold(dist, dist.max()*0.6, 255, cv.THRESH_BINARY)  # 找到marks
    cv.imshow('surfacebi', surface)
    surface_fg = np.uint8(surface)
    unknown = cv.subtract(sure_bg, surface_fg)
    ret, markers = cv.connectedComponents(surface_fg)
    print(ret)
    markers = markers + 1
    markers[unknown == 255] = 0  # 像素操作
    markers = cv.watershed(src, markers=markers)
    src[markers == -1] = [0, 0, 255]
    cv.imshow('res', src)


"""人脸检测，"""
"""下载地址haar和lbp脸部特征数据：https://github.com/opencv/opencv/tree/master/data"""
#  -*- encode: utf-8 -*-
def face_detector_demo():  # 图片人脸检测
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier('e:/cv33/haarcascade_frontalface_alt_tree.xml')
    faces = face_detector.detectMultiScale(gray, 1.02, 5)
    for x, y, w, h in faces:
        cv.rectangle(src, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv.imshow('result', src)
src = cv.imread('e:/cv33/lena.png')
cv.namedWindow('result', cv.WINDOW_AUTOSIZE)
cv.imshow('original', src)
face_detector_demo()
cv.waitKey(0)


def video_face_demo(image):  # 视频人脸检测
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier('e:/cv33/haarcascade_frontalface_alt_tree.xml')
    faces = face_detector.detectMultiScale(gray, 1.1, 2)
    for x, y, w, h in faces:
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv.imshow('result', image)
capture = cv.VideoCapture('e:/cv33/tr/you.mp4')
while True:
    ret, frame = capture.read()
    frame = cv.flip(frame, 1)
    video_face_demo(frame)
    c = cv.waitKey(10)
    if c == 27:
        break
cv.destroyAllWindows()











