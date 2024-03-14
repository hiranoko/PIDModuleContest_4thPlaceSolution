import numpy as np
from cv2 import COLOR_HSV2RGB, cvtColor


def diamond_square(n, decay=0.5, fixed_corners=True):
    s = 2**n + 1
    a = np.zeros((s, s), dtype=np.float64)

    if fixed_corners:
        a[0 :: s - 1, 0 :: s - 1] = 0.5
    else:
        a[0 :: s - 1, 0 :: s - 1] = np.random.rand(2, 2)

    for k in range(1, n + 1):
        m = 0.5 * np.exp(decay * (1 - k))
        ss = s // (2**k)
        ni = 2**k

        # Diamond step
        ru, cl = np.meshgrid(
            np.arange(0, ni, 2) * ss, np.arange(0, ni, 2) * ss, indexing="ij"
        )
        r, c = ru + ss, cl + ss
        rd, cr = r + ss, c + ss

        a[r, c] = 0.25 * (a[ru, cl] + a[ru, cr] + a[rd, cl] + a[rd, cr])
        a[r, c] += np.random.uniform(-m, m, a[r, c].shape)

        # Square step
        r_idx = np.arange(ni) * ss
        c_idx = np.arange(ni) * ss

        ru = np.where(r_idx > 0, r_idx - ss, s - ss - 1)
        rd = np.where(r_idx < s - 1, r_idx + ss, ss)

        for i, r in enumerate(r_idx):
            sj = 1 if i % 2 == 0 else 0
            c_idx_filtered = c_idx[sj::2]

            cl = np.where(c_idx_filtered > 0, c_idx_filtered - ss, s - ss - 1)
            cr = np.where(c_idx_filtered < s - 1, c_idx_filtered + ss, ss)

            a[r, c_idx_filtered] = 0.25 * (
                a[ru[i], c_idx_filtered]
                + a[r, cl]
                + a[r, cr]
                + a[rd[i], c_idx_filtered]
            )
            a[r, c_idx_filtered] += np.random.uniform(-m, m, c_idx_filtered.shape)
    return a


def _colorize(ds):
    img = np.zeros((ds.shape[0], ds.shape[1], 3), dtype=np.uint8)

    scales = np.random.uniform([0.25, 0.1, 0.3], [1, 0.3, 0.6]) * 255
    shifts = np.random.uniform([0, 0.4, 0.4], [1, 0.6, 0.6]) * 255

    img = np.clip(ds[:, :, np.newaxis] * scales + shifts, 0, 255).astype(np.uint8)

    return img


def colorized_ds(size=256):
    n = int(np.ceil(np.log2(size)))
    r = diamond_square(n, np.random.uniform(0.4, 0.8), fixed_corners=False)[
        :size, :size
    ]
    img = _colorize(r)
    img = cvtColor(img, COLOR_HSV2RGB, dst=img)
    return img
