import os
import pickle
import random
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from _fractal import centoroid, generate_point, generator, min_max
from _point_cloud import (
    generate_colors_with_alpha,
    random_rotate_point_cloud,
    random_sample_point_cloud,
    random_shift_scale_point_cloud,
)
from albumentations import (
    Compose,
    GridDistortion,
    HorizontalFlip,
    HueSaturationValue,
    OneOf,
    OpticalDistortion,
    RandomBrightnessContrast,
    ShiftScaleRotate,
    Transpose,
    VerticalFlip,
)
from pytorch3d.renderer import (
    AlphaCompositor,
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    look_at_view_transform,
)
from pytorch3d.structures import Pointclouds
from tqdm import tqdm


class args:
    numof_class = 1000
    numof_instance = 1000
    start_numof_classes = 0
    seed = 42
    point_num = 10000
    variance = 0.05
    normalize = 1.0
    param_file_name = "point_cloud_C1000_P10000_params.pkl"


def generate_fractal(args: args):
    param_size = np.random.randint(2, 8)
    params = np.zeros((param_size, 13), dtype=float)
    sum_proba = 0.0

    for m in range(param_size):
        param_rand = np.random.uniform(-1.0, 1.0, 12)
        a, b, c, d, e, f, g, h, i, j, k, l = param_rand[0:12]
        check_param = np.array(param_rand[0:9], dtype=float).reshape(3, 3)
        prob = abs(np.linalg.det(check_param))
        sum_proba += prob
        params[m, 0:13] = a, b, c, d, e, f, g, h, i, j, k, l, prob

    for m in range(param_size):
        params[m, 12] /= sum_proba

    fractal_point = generator(params, args.point_num)
    point = min_max(fractal_point, args.normalize, axis=None)
    point = centoroid(point, args.point_num)
    var_point = np.var(point, axis=1)
    arr = np.isnan(point).any(axis=1)
    valid = not arr[1] and all(v > args.variance for v in var_point)
    return valid, point, params


class Generator:
    @classmethod
    def get_params(cls, param_root_path: str):
        param_pkl_path = os.path.join(param_root_path, args.param_file_name)
        if os.path.exists(param_pkl_path):
            with open(param_pkl_path, "rb") as f:
                cls.ifs_params = pickle.load(f)
        else:
            random.seed(args.seed)
            np.random.seed(args.seed)
            ifs_params = {}
            for idx in tqdm(
                range(args.start_numof_classes, args.numof_class),
                leave=False,
                total=(args.numof_class - args.start_numof_classes),
            ):
                valid = False
                while not valid:
                    valid, _, params = generate_fractal(args)
                ifs_params[idx] = params
            cls.ifs_params = ifs_params
            # raise Exception("params.pkl is not found.")

    @classmethod
    def generate(cls, out_path: str):
        # Set device
        assert torch.cuda.is_available(), "GPU is not available."
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set random seed
        random.seed(args.seed)
        np.random.seed(args.seed)

        # Set ifs parameters
        ifs_params = cls.ifs_params

        # Generate point cloud
        point_clouds = {}
        for idx in tqdm(range(len(ifs_params))):
            point_data = generate_point(ifs_params[idx], args.point_num)
            point_clouds[idx] = point_data.transpose()

        # Setting of Rendering
        R, T = look_at_view_transform(10.0, 0.0, 0.0)
        rasterizer = PointsRasterizer(
            cameras=FoVPerspectiveCameras(device=device, R=R, T=T, fov=10.0),
            raster_settings=PointsRasterizationSettings(
                image_size=384, radius=0.015, points_per_pixel=4
            ),
        )
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color=(0.0, 0.0, 0.0)),
        )

        # Augmentation
        transfrom = Compose(
            [
                Transpose(),
                HorizontalFlip(),
                VerticalFlip(),
                ShiftScaleRotate(),
                RandomBrightnessContrast(),
                HueSaturationValue(),
                OneOf([OpticalDistortion(), GridDistortion()]),
            ]
        )

        for cls_idx in tqdm(range(args.numof_class)):
            save_dir = Path(out_path) / str(cls_idx).zfill(5)
            save_dir.mkdir(parents=True, exist_ok=True)

            saved_images = []
            for img_idx in tqdm(range(args.numof_instance), leave=False):
                points_3d = point_clouds[cls_idx]

                # Augmentation
                points_3d = random_sample_point_cloud(points_3d, 1.0)
                points_3d = random_shift_scale_point_cloud(points_3d, 0.5)
                points_3d = random_rotate_point_cloud(points_3d, 1.0)

                # Colors
                colors = generate_colors_with_alpha(points_3d)

                # Make Pointclouds
                points_3d = torch.from_numpy(points_3d.astype(np.float32))
                verts = torch.Tensor(points_3d).to(device)
                rgba = torch.Tensor(colors).to(device)
                point_cloud = Pointclouds(points=[verts], features=[rgba])

                # Render
                render_images = renderer(point_cloud)
                saved_images.append(
                    (render_images[0] * 255).cpu().numpy().astype(np.uint8)
                )

            for img_idx in range(args.numof_instance):
                save_name = save_dir / (str(img_idx).zfill(5) + ".png")
                augmented = transfrom(image=saved_images[img_idx][:, :, :3])
                image = augmented["image"]
                cv2.imwrite(str(save_name), image)
