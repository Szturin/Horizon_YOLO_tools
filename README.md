## 文件结构
- number.bin 板端汇编模型
- cv_main.py 主程序（注意：使用TCP进行推流，需要在主机运行客户端程序）
## 推理部分
模型推理过程在BPU上运行，与CPU的量化和反量化并行执行，耗时约30ms~40ms

## 后处理部分
模型后处理过程在CPU上运行，使用Cython编译C++代码为Python接口，便于Python主程序调用，耗时约20ms~35ms

## 板端实时模型运行帧数
采用TCP协议编解码进行视频推流，约10~15帧，BPU利用率不高，程序虽然采用推理+后处理的双线程，但是几乎没有起到效果。
但是采用TROS接口部署，依然使用本例模型，BPU利用率双核均跑满，帧数跑满30fps(1080p摄像头最高帧率)，所以模型理论上不存在问题。
关于多线程技术，可能需要进一步深究
