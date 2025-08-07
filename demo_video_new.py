import copy
import numpy as np
import cv2
import os
import json
import subprocess
from pathlib import Path
from typing import NamedTuple, List, Dict, Optional
from src import util
from src.body import Body
from src.hand import Hand
import ffmpeg

# 初始化模型
body_estimation = Body('/root/autodl-tmp/pytorch-openpose-master/model/body_pose_model.pth')
hand_estimation = Hand('/root/autodl-tmp/pytorch-openpose-master/model/hand_pose_model.pth')

# 视频处理配置
INPUT_VIDEO = "/root/autodl-tmp/3.mp4"  # 固定输入路径
OUTPUT_DIR = "/root/autodl-tmp/pose_frames/"  # 帧序列输出目录
OUTPUT_VIDEO = "/root/autodl-tmp/3_processed.mp4"  # 处理后的视频路径
PROCESS_BODY = True  # 是否处理身体姿态
PROCESS_HANDS = True  # 是否处理手部姿态
SAVE_KEYPOINTS = True  # 是否保存关键点数据

class FFProbeResult(NamedTuple):
    return_code: int
    json: str
    error: str

class KeypointData(NamedTuple):
    frame: int
    body: Optional[List[List[float]]]
    hands: Optional[List[List[List[float]]]]

def ffprobe(file_path) -> FFProbeResult:
    """获取视频元信息"""
    command_array = ["ffprobe", "-v", "quiet", "-print_format", "json",
                    "-show_format", "-show_streams", file_path]
    result = subprocess.run(command_array, stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE, universal_newlines=True)
    return FFProbeResult(return_code=result.returncode,
                        json=result.stdout,
                        error=result.stderr)

def process_frame(frame: np.ndarray, body: bool = True, hands: bool = True) -> tuple:
    """处理单帧图像并返回处理后的图像和关键点数据"""
    canvas = copy.deepcopy(frame)
    body_keypoints = None
    hand_keypoints = []
    
    # 调整图像尺寸以优化处理[6](@ref)
    frame_resized = cv2.resize(frame, (368, 368))  # OpenPose推荐尺寸[4](@ref)
    
    if body:
        try:
            candidate, subset = body_estimation(frame_resized)
            if candidate is not None and subset is not None:
                # 将关键点坐标映射回原始图像尺寸
                h_ratio = frame.shape[0] / frame_resized.shape[0]
                w_ratio = frame.shape[1] / frame_resized.shape[1]
                candidate[:, 0] *= w_ratio
                candidate[:, 1] *= h_ratio
                
                canvas = util.draw_bodypose(canvas, candidate, subset)
                body_keypoints = candidate.tolist()
                
                if hands:
                    hands_list = util.handDetect(candidate, subset, frame_resized)
                    for x, y, w, is_left in hands_list:
                        # 确保手部区域在图像范围内
                        x, y, w = int(x*w_ratio), int(y*h_ratio), int(w*max(w_ratio, h_ratio))
                        if y + w > frame.shape[0] or x + w > frame.shape[1]:
                            continue
                            
                        hand_roi = frame[y:y+w, x:x+w]
                        if hand_roi.size == 0:
                            continue
                            
                        peaks = hand_estimation(hand_roi)
                        if peaks is not None:
                            peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
                            peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
                            hand_keypoints.append(peaks.tolist())
                            canvas = util.draw_handpose(canvas, [peaks])
        except Exception as e:
            print(f"姿态估计出错: {str(e)}")
    
    return canvas, KeypointData(frame=0, body=body_keypoints, hands=hand_keypoints if hand_keypoints else None)

def save_keypoints(output_dir: str, frame_count: int, keypoints: KeypointData):
    """保存关键点数据到JSON文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    keypoints_dict = {
        'frame': frame_count,
        'body': keypoints.body,
        'hands': keypoints.hands
    }
    
    with open(f"{output_dir}/keypoints_{frame_count:05d}.json", 'w') as f:
        json.dump(keypoints_dict, f, indent=2)

class VideoWriter:
    """视频写入器（FFmpeg实现）"""
    def __init__(self, output_file: str, fps: float, frame_size: tuple, pix_fmt: str = "bgr24", vcodec: str = "libx264"):
        if os.path.exists(output_file):
            os.remove(output_file)
        self.ff_proc = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt=pix_fmt,
                  s=f'{frame_size[0]}x{frame_size[1]}', r=fps)
            .output(output_file, pix_fmt=pix_fmt, vcodec=vcodec)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
    
    def write(self, frame: np.ndarray):
        self.ff_proc.stdin.write(frame.tobytes())
    
    def close(self):
        self.ff_proc.stdin.close()
        self.ff_proc.wait()

def save_frame(frame: np.ndarray, output_dir: str, frame_count: int):
    """保存单帧到图片序列"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(f"{output_dir}/frame_{frame_count:05d}.jpg", frame)

def main():
    # 获取视频信息
    probe_result = ffprobe(INPUT_VIDEO)
    info = json.loads(probe_result.json)
    video_info = next(s for s in info["streams"] if s["codec_type"] == "video")
    
    # 初始化视频读取
    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps = eval(video_info["avg_frame_rate"])  # 处理分数形式的帧率（如30/1）
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 初始化写入器
    writer = None
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理当前帧
            processed_frame, keypoints = process_frame(frame, PROCESS_BODY, PROCESS_HANDS)
            
            # 初始化写入器（第一帧时）
            if writer is None:
                frame_size = processed_frame.shape[:2][::-1]  # 注意OpenCV是(width, height)
                writer = VideoWriter(OUTPUT_VIDEO, fps, frame_size)
            
            # 保存帧序列、关键点和视频
            save_frame(processed_frame, OUTPUT_DIR, frame_count)
            if SAVE_KEYPOINTS:
                save_keypoints(OUTPUT_DIR, frame_count, keypoints._replace(frame=frame_count))
            writer.write(processed_frame)
            
            frame_count += 1
            print(f"\r处理进度: {frame_count}/{total_frames} ({frame_count/total_frames:.1%})", end="")
    
    finally:
        cap.release()
        if writer:
            writer.close()
        print(f"\n处理完成！\n视频已保存到: {OUTPUT_VIDEO}\n帧序列和关键点数据已保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()