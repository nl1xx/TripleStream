import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import DualTransform
from scipy.ndimage import gaussian_filter

class FaceKeypointBBoxSafeCrop(DualTransform):
    """
    基于面部关键点的安全裁剪
    """
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def get_bbox(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            return (0, 0, image.shape[1], image.shape[0])
        x, y, w, h = faces[0]
        return (x, y, x+w, y+h)

    def apply(self, img, **params):
        x_min, y_min, x_max, y_max = self.get_bbox(img)
        return img[y_min:y_max, x_min:x_max]

class TemporalElasticTransform:
    """
    时序弹性变换
    """
    def __init__(self, alpha=20, sigma=5, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, frames):
        if np.random.rand() > self.p:
            return frames
        
        # 生成随机位移场
        displacement = np.random.uniform(-self.alpha, self.alpha, size=len(frames))
        displacement = gaussian_filter(displacement, sigma=self.sigma)
        
        # 应用插值
        new_frames = []
        for t in range(len(frames)):
            src_t = min(max(int(t + displacement[t]), 0), len(frames)-1)
            new_frames.append(frames[src_t])
        return new_frames


class MotionMagnification:
    """
    基于相位的光流运动放大（适配灰度图输入）
    """

    def __init__(self, factor=1.2):
        self.factor = factor

    def __call__(self, frames):
        if len(frames) < 2:
            return frames

        # 输入已经是灰度图（单通道）
        prev = frames[0]
        mag_frames = [prev]
        for frame in frames[1:]:
            curr = frame  # 直接使用单通道输入

            # 计算光流
            flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # 对单通道图像直接做remap（无需通道操作）
            magnified = cv2.remap(
                curr,  # 使用当前帧作为源
                flow[..., 0] * self.factor,
                flow[..., 1] * self.factor,
                interpolation=cv2.INTER_LINEAR
            )

            mag_frames.append(magnified)
            prev = curr  # 更新前一帧

        return mag_frames
