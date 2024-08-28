import numpy as np
import cv2
from hobot_dnn import pyeasy_dnn as dnn
import time
import os
import serial
import threading  # 使用 threading 模块进行多线程操作
import yolov5_post
import colorsys
from server import VideoStreaming
import hobot_vio.libsrcampy as srcampy

# 框颜色设置
colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

# 获取显示器高度和宽度
def get_display_res(self):
    if not os.path.exists("/usr/bin/get_hdmi_res"):
        return 1920, 1080

    import subprocess
    p = subprocess.Popen(["/usr/bin/get_hdmi_res"], stdout=subprocess.PIPE)
    result = p.communicate()
    res = result[0].split(b',')
    res[1] = max(min(int(res[1]), 1920), 0)
    res[0] = max(min(int(res[0]), 1080), 0)
    return int(res[1]), int(res[0])   

#hdmi输出初始化
disp = srcampy.Display()
disp_w, disp_h = get_display_res(disp)

# 摄像头类创建
class pi_Camera():
    # 类的初始化
    def __init__(self):
        # 图像初始化配置
        self.Video = None
        self.cached_w, self.cached_h = None, None

        # 遍历摄像头设备号
        for index in range(7, 21):
            self.Video = cv2.VideoCapture(index)
            if self.Video.isOpened():
                print(f"The video{index} is opened.")
                break
            else:
                print(f"No video{index}.")
        
        # 如果没有有效的摄像头则抛出异常
        if not self.Video or not self.Video.isOpened():
            raise Exception("No valid video capture device found.")

        # 设置视频参数
        self.Video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.Video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)        # 列 宽度
        self.Video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)       # 行 高度
    
    # HDMI图形显示
    def hdmi_show(self, image):
        global disp_w, disp_h
        try:
            print(f"Begin HDMI show with resolution {image.shape[1]}x{image.shape[0]}")
            
            if self.cached_w != disp_w or self.cached_h != disp_h:
                print(f"Display resolution changed: {disp_w}x{disp_h}")
                self.cached_w, self.cached_h = disp_w, disp_h
            
            if image.shape[0] != disp_h or image.shape[1] != disp_w:
                print(f"Resizing image from {image.shape[1]}x{image.shape[0]} to {disp_w}x{disp_h}")
                image = cv2.resize(image, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            
            nv12_frame = bgr2nv12_opencv(image)
            print(f"NV12 frame size: {nv12_frame.size}, expected size: {(disp_h + disp_h // 2) * disp_w}")
            
            if nv12_frame.size != (disp_h + disp_h // 2) * disp_w:
                print(f"Error: NV12 frame size {nv12_frame.size} does not match expected size {(disp_h + disp_h // 2) * disp_w}")
            else:
                disp.set_img(nv12_frame.tobytes())
            
            print("Image set to HDMI display")
        except Exception as e:
            print(f"Error in hdmi_show: {str(e)}")

# ROI 处理器类创建
class ROIPixelProcessor:
    def __init__(self):
        pass
    
    # 处理ROI
    def process_roi(self, image, roi):
        """
        取得 ROI 区域的像素值进行处理，但不改变索引号
        :param image: 输入图像
        :param roi: ROI 参数 (x, y, width, height)
        :return: ROI 区域的像素值
        """
        x, y, w, h = roi
        
        # 确保 ROI 在图像范围内
        if x < 0:
            w += x
            x = 0
        if y < 0:
            h += y
            y = 0
        if x + w > image.shape[1]:
            w = image.shape[1] - x
        if y + h > image.shape[0]:
            h = image.shape[0] - y
        
        # 提取 ROI 区域的像素值
        roi_pixels = image[y:y+h, x:x+w]
        return roi_pixels

# 将BGR图像转换为NV12格式
def bgr2nv12_opencv(image):
    height, width = image.shape[:2]
    # Convert BGR to YUV420p
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
    
    y = yuv420p[:height, :width]
    uv = yuv420p[height:, :].flatten()
    
    # Create NV12 image
    nv12 = np.zeros((height + height // 2, width), dtype=np.uint8)
    nv12[:height] = y
    nv12[height:] = uv.reshape(-1, width)
    
    return nv12

def get_classes(class_file_name='coco_classes.names'):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def draw_bboxs(image, bboxes, gt_classes_index=None, classes=get_classes()):
    """draw the bboxes in the original image
    """
    num_classes = len(classes)
    image_h, image_w, channel = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))

    fontScale = 0.5
    bbox_thick = int(0.6 * (image_h + image_w) / 600)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)

        if gt_classes_index == None:
            class_index = int(bbox[5])
            score = bbox[4]
        else:
            class_index = gt_classes_index[i]
            score = 1

        bbox_color = colors[class_index]
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
        classes_name = classes[class_index]
        bbox_mess = '%s: %.2f' % (classes_name, score)
        t_size = cv2.getTextSize(bbox_mess,
                                 0,
                                 fontScale,
                                 thickness=bbox_thick // 2)[0]
        cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3),
                      bbox_color, -1)
        cv2.putText(image,
                    bbox_mess, (c1[0], c1[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale, (0, 0, 0),
                    bbox_thick // 2,
                    lineType=cv2.LINE_AA)
        print("{} is in the picture with confidence:{:.4f}".format(
            classes_name, score))
    return image

# 获取张量的高度和宽度
def get_hw(pro):
    if pro.layout == "NCHW":
        return pro.shape[2], pro.shape[3]
    else:
        return pro.shape[1], pro.shape[2]

# 打印张量属性
def print_properties(pro):
    print("tensor type:", pro.tensor_type)
    print("data type:", pro.dtype)
    print("layout:", pro.layout)
    print("shape:", pro.shape)

# 定义函数，计算目标框
def compute_box(out, size, cls_num, img_w, img_h, original_w, original_h): 
    # 调用yolov5_post模块中的postprocess函数进行后处理
    results = yolov5_post.postprocess(out, size[0], cls_num, img_w, img_h)
    # 进行坐标的比例缩放调整
    for result in results:
        result[0] = int(result[0] * original_w / img_w)  # x_min
        result[1] = int(result[1] * original_h / img_h)  # y_min
        result[2] = int(result[2] * original_w / img_w)  # x_max
        result[3] = int(result[3] * original_h / img_h)  # y_max
    return results

host = '192.168.123.175'
port = 8080
streamer = VideoStreaming(host, port)

number_start = 0  # 初始化数字

Watch = pi_Camera()  # 摄像头模块

# 串口初始化
ser = serial.Serial('/dev/ttyS3', 115200, timeout=1)  # 1s timeout
print(ser)
print("Starting demo now! Press CTRL+C to exit")

started = time.time()
last_logged = time.time()
frame_count = 0

# 开启CPU性能模式
os.system("sudo bash -c 'echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor'")
os.system("sudo bash -c 'echo 1 > /sys/devices/system/cpu/cpufreq/boost'")

# 加载模型
models = dnn.load('number_2024.bin')
# 打印模型信息
for output in models[0].outputs:
    print_properties(output.properties)

# 多线程管理
def inference_thread(model, nv12_data, result_holder):
    result_holder['inference'] = model.forward(nv12_data)
    print(f"推理完成，耗时: {time.time() - t0:.6f} 秒")

def postprocess_thread(result_holder, img_file):
    result_holder['postprocess'] = compute_box(result_holder['inference'], (672, 672), 8, 672, 672, img_file.shape[1], img_file.shape[0])
    print(f"后处理完成，耗时: {time.time() - t0:.6f} 秒")

try:
    while True:
        # 读取图像文件
        ret, img_file = Watch.Video.read()
        if not ret:
            print("Error: Unable to read from video capture")
            break
        
        # 调整图像大小
        resized_data = cv2.resize(img_file, (672, 672), interpolation=cv2.INTER_AREA)
        # 转换图像格式
        nv12_data = bgr2nv12_opencv(resized_data)
        
        t0 = time.time()

        # 初始化线程结果存储字典
        result_holder = {}

        # 创建并启动推理线程
        inference_thread_obj = threading.Thread(target=inference_thread, args=(models[0], nv12_data, result_holder))
        inference_thread_obj.start()

        # 等待推理完成
        inference_thread_obj.join()

        # 创建并启动后处理线程
        postprocess_thread_obj = threading.Thread(target=postprocess_thread, args=(result_holder, img_file))
        postprocess_thread_obj.start()

        # 等待后处理完成
        postprocess_thread_obj.join()

        # 绘制检测框并显示图像
        img_with_bboxes = draw_bboxs(img_file, result_holder['postprocess'])
        t3 = time.time()
        print(f"draw box time: {t3 - t0:.6f} 秒")

        # cv2.imshow("result",img_file)
        streamer.send(img_file)
        # 帧数计算
        frame_count += 1
        now = time.time()

        # 每秒计算一次帧率
        if now - last_logged >= 1.0:
            fps = frame_count / (now - last_logged)  # 计算帧率
            print(f"{fps:.2f} fps")  # 打印帧率
            last_logged = now  # 更新 last_logged 时间戳
            frame_count = 0  # 重置 frame_count

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    Watch.Video.release()  # 释放视频资源
    cv2.destroyAllWindows()  # 关闭所有窗口