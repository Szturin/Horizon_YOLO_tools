#!/usr/bin/env python3
import sys
import os
import time
import my_serial


def list_uart_devices():
    devices = os.popen('ls /dev/tty[a-zA-Z]*').read().split()
    for idx, device in enumerate(devices, 1):
        print(f"{idx}: {device}")
    return devices

def prompt_for_uart_device(devices):
    while True:
        try:
            choice = int(input("请选择需要测试的串口设备编号: ")) - 1
            if 0 <= choice < len(devices):
                uart_dev = devices[choice]
                break
            else:
                print("无效的编号，请重新选择。")
        except ValueError:
            print("请输入有效的数字。")
    baudrate = input("请输入波特率(9600,19200,38400,57600,115200,921600):")
    return uart_dev, baudrate

def Serial_config():
    # 列出可用的 UART 设备
    devices = list_uart_devices()
    
    # 让用户选择设备
    uart_dev, baudrate = prompt_for_uart_device(devices)
    
    # 设置串口
    try:
        ser = serial.Serial(uart_dev, int(baudrate), timeout=1) # 1s timeout
        print(ser)
        print("Starting demo now! Press CTRL+C to exit")
    except serial.SerialException as e:
        print(f"无法打开串口 {uart_dev}: {e}")

if __name__=='__main__':
    # 调用函数进行配置
    Serial_config()