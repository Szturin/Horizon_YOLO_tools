import numpy as np
import cv2
import socket
import struct
import threading

class VideoStreaming(object):
    def __init__(self, host, port):
        # 创建一个TCP/IP Socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 设置Socket选项，允许端口重用
        self.server_socket.bind((host, port))  # 绑定主机IP地址和端口号
        self.server_socket.listen(5)  # 设置监听数量

        print(f"Host: {host}")
        print("等待客服端连接...")

        self.connection, self.client_address = self.server_socket.accept()  # 等待客户端连接
        self.connect = self.connection.makefile('wb')  # 创建一个传输文件写入数据
        print(f"连接成功：{self.client_address}")

    def send(self, _img):
        """发送图像数据
        ----
        * _img: 传入的图像数据"""
        try:
            # 编码图像并降低质量以减少数据量
            _, img_encode = cv2.imencode('.jpg', _img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])  # 使用50的JPEG质量压缩
            data_encode = img_encode.tobytes()  # 将编码图像转换为字节数据
            # 使用struct.pack发送图像长度信息，然后发送图像数据
            self.connect.write(struct.pack('<L', len(data_encode)) + data_encode)
            self.connect.flush()  # 刷新输出流
        except Exception as e:
            print(f"发送失败: {e}")
            self.connect.close()  # 关闭连接

    def close(self):
        """关闭连接"""
        self.connect.close()
        self.server_socket.close()

if __name__ == '__main__':
    host = '192.168.96.34'  # 树莓派的IP地址
    port = 8000  # 端口号

    cap = cv2.VideoCapture(0)  # 使用默认摄像头

    streamer = VideoStreaming(host, port)

    while True:
        ret, frame = cap.read()  # 读取摄像头帧
        if not ret:
            break
        streamer.send(frame)  # 发送帧数据

    cap.release()  # 释放摄像头资源
    streamer.close()  # 关闭服务端连接