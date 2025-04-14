import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from my_transforms import FaceKeypointBBoxSafeCrop

# CASME II原始数据结构配置
RAW_DATA_ROOT = "/path/to/CASME2_RAW"  
PROCESSED_ROOT = "/path/to/CASME2_processed" 
META_CSV = "CASME2_meta.csv" 
CLASS_MAP = {
    "happiness": 0,
    "surprise": 1,
    "disgust": 2,
    "repression": 3,
    "others": 4
}

def load_metadata():
    """
    加载CASME II元数据
    """
    df = pd.read_excel(os.path.join(RAW_DATA_ROOT, "CASME2-Appendix.xlsx"))
    df = df[['Subject', 'Filename', 'Onset', 'Apex', 'Offset', 'Estimated Emotion']]
    df['Emotion'] = df['Estimated Emotion'].apply(lambda x: "others" if pd.isna(x) else x.lower())
    return df

def extract_faces(video_path, output_dir, label, face_detector):
    """
    从单个视频中提取面部ROI帧
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success = True
    
    while success:
        success, frame = cap.read()
        if not success:
            break
        
        # 转换颜色空间
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 面部检测与裁剪
        try:
            cropped = face_detector.apply(frame_rgb)
        except:
            cropped = frame_rgb  # 失败时使用原帧
        
        # 保存处理后的帧
        output_path = os.path.join(output_dir, f"{label}_{frame_count:04d}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
        frame_count += 1
    
    cap.release()
    return frame_count

def main():
    # 创建输出目录结构
    Path(PROCESSED_ROOT).mkdir(parents=True, exist_ok=True)
    for cls in CLASS_MAP.keys():
        Path(os.path.join(PROCESSED_ROOT, cls)).mkdir(exist_ok=True)
    
    # 初始化面部检测器
    face_detector = FaceKeypointBBoxSafeCrop()
    
    # 加载元数据
    meta_df = load_metadata()
    print(f"共加载 {len(meta_df)} 个样本元数据")
    
    # 处理每个视频
    for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
        # 构建输入路径
        video_path = os.path.join(
            RAW_DATA_ROOT, 
            "CASME2_RAW", 
            row['Subject'], 
            row['Filename'] + ".avi"
        )
        
        if not os.path.exists(video_path):
            print(f"警告：视频文件 {video_path} 不存在")
            continue
        
        # 创建输出目录
        output_dir = os.path.join(PROCESSED_ROOT, row['Emotion'])
        Path(output_dir).mkdir(exist_ok=True)
        
        # 提取面部帧
        frame_count = extract_faces(
            video_path, 
            output_dir, 
            row['Filename'], 
            face_detector
        )
        
        print(f"已处理 {row['Filename']}，提取 {frame_count} 帧")

if __name__ == "__main__":
    main()
