import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def random_rotate_point_cloud(points_3d: np.ndarray) -> np.ndarray:
    rotation_matrix = Rotation.random().as_matrix()
    euler_angles = Rotation.from_matrix(rotation_matrix).as_euler("xyz", degrees=True)
    rotated_points = np.dot(points_3d, rotation_matrix.T)
    return rotated_points, euler_angles


def rotate_point_cloud(points_3d: np.ndarray, euler_angles: np.ndarray) -> np.ndarray:
    rotation_matrix = Rotation.from_euler("xyz", euler_angles, degrees=True).as_matrix()
    rotated_points = np.dot(points_3d, rotation_matrix.T)
    return rotated_points


def render_point_cloud(points_3d: np.ndarray, image_size: int) -> np.ndarray:
    points_2d = points_3d[:, :2] * image_size / 2 + image_size / 2
    image = np.zeros((image_size, image_size), dtype=np.uint8)
    for x, y in points_2d:
        cv2.circle(
            image, (int(round(y)), int(round(x))), radius=1, color=255, thickness=-1
        )
    return image


def random_shift_point_cloud(
    point_cloud: np.ndarray, shift_value: float = 0.2, prob: float = 0.5
) -> np.ndarray:
    if np.random.random() < prob:
        shift = np.random.uniform(
            -shift_value, shift_value, size=point_cloud.shape[1]
        )  # シフトはランダムに
        point_cloud = point_cloud + shift
    return point_cloud


def random_scale_point_cloud(
    point_cloud: np.ndarray, scale_value: float = 0.2, prob: float = 0.5
) -> np.ndarray:
    if np.random.random() < prob:
        scale = np.random.uniform(
            -scale_value, scale_value, size=point_cloud.shape[1]
        )  # シフトはランダムに
        point_cloud = point_cloud * scale
    return point_cloud


def random_shift_scale_point_cloud(
    point_cloud: np.ndarray, prob: float = 0.5
) -> np.ndarray:
    if np.random.random() < prob:
        scale = np.random.uniform(0.8, 1.2)  # スケールは適当な範囲で調整
        shift = np.random.uniform(-0.2, 0.2, size=point_cloud.shape[1])  # シフトはランダムに
        point_cloud = point_cloud * scale + shift
    return point_cloud


def random_rotate_point_cloud(point_cloud: np.ndarray, prob: float = 0.5) -> np.ndarray:
    if np.random.random() < prob:
        rotation_matrix = Rotation.random().as_matrix()
        point_cloud = np.dot(point_cloud, rotation_matrix.T)
    return point_cloud


def random_rotate_point_cloud_axis_specific(
    point_cloud: np.ndarray, axis: int = 2, prob: float = 0.5
) -> np.ndarray:
    if np.random.random() < prob:
        # 軸に応じたランダムな回転角度を生成
        angles = np.zeros(3)
        angles[axis] = np.random.uniform(0, 2 * np.pi)

        # 回転行列を生成
        rotation_matrix = Rotation.from_euler("xyz", angles, degrees=False).as_matrix()

        # ポイントクラウドを回転
        point_cloud = np.dot(point_cloud, rotation_matrix.T)

    return point_cloud


def random_sample_point_cloud(
    points_3d: np.ndarray, min_ratio: float = 0.8, prob: float = 0.5
) -> np.ndarray:
    if np.random.random() < prob:
        sample_num = int(points_3d.shape[0] * np.random.uniform(min_ratio, 1.0))
        indices = np.random.choice(points_3d.shape[0], sample_num, replace=False)
        points_3d = points_3d[indices]
    return points_3d


def random_gaussian_noise_point_cloud(
    points_3d: np.ndarray, prob: float = 0.5
) -> np.ndarray:
    if np.random.random() < prob:
        noise = np.random.normal(0, 0.01, size=points_3d.shape)
        points_3d = points_3d + noise
    return points_3d


def random_flip_point_cloud(points_3d: np.ndarray, prob: float = 0.5) -> np.ndarray:
    if np.random.random() < prob:
        points_3d[:, 0] = -points_3d[:, 0]
    return points_3d


def generate_random_colors(points):
    """
    各点に対してランダムな色を生成する。
    - points: 点群データ (N, 3)
    """
    # 0~1の乱数を生成
    colors = np.random.rand(points.shape[0], 3)
    return colors


def generate_colors(points):
    """
    各点に対して透明度を含む色を生成する。
    - points: 点群データ (N, 3)
    - alpha: アルファ値 (透明度)
    """
    # RGBは1で統一し、アルファ値を最後に追加
    colors = np.ones((points.shape[0], 3), dtype=np.float32)
    return colors


def generate_colors_with_alpha(points, alpha=0.5):
    """
    各点に対して透明度を含む色を生成する。
    - points: 点群データ (N, 3)
    - alpha: アルファ値 (透明度)
    """
    # RGBは1で統一し、アルファ値を最後に追加
    colors = np.ones((points.shape[0], 4), dtype=np.float32)
    colors[:, -1] = alpha  # アルファ値の設定
    return colors


def colorize_point_cloud(points, min_sat=0.3, min_val=0.5, random_mode=True):
    """
    点群データに色をつける。Z座標に基づいて色相を決定。彩度と明度はランダムモードまたは確定的に選ぶ。

    Args:
        points (np.ndarray): 点群データ。形状は(N, 3)。
        min_sat (float): 彩度の最小値。
        min_val (float): 明度の最小値。
        random_mode (bool): ランダムモードの使用有無。

    Returns:
        np.ndarray: 色付きの点群データ。形状は(N, 3)。
    """
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    colors = np.empty((points.shape[0], 3), dtype=np.uint8)

    if random_mode:
        # ランダムモード
        sat = np.random.uniform(min_sat, 1) * 255
        val = np.random.uniform(min_val, 1) * 255
    else:
        # 確定的モード
        sat = min_sat * 255
        val = min_val * 255

    for i, point in enumerate(points):
        hue = np.uint8((point[2] - z_min) / (z_max - z_min) * 255)
        colors[i] = [hue, sat, val]

    colors = cv2.cvtColor(colors.reshape(1, -1, 3), cv2.COLOR_HSV2RGB).reshape(-1, 3)
    colors = colors.astype(np.float32) / 255

    return colors
