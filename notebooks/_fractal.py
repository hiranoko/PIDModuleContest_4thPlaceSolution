import numpy as np


def generator(params, point_num: int):
    generators = ifs_function()
    for param in params:
        generators.set_param(
            float(param[0]),
            float(param[1]),
            float(param[2]),
            float(param[3]),
            float(param[4]),
            float(param[5]),
            float(param[6]),
            float(param[7]),
            float(param[8]),
            float(param[9]),
            float(param[10]),
            float(param[11]),
            float(param[12]),
        )
    data = generators.calculate(point_num)
    return data


def min_max(x: np.ndarray, normalize: float, axis=None):
    min = np.min(x, axis=axis, keepdims=True)
    max = np.max(x, axis=axis, keepdims=True)
    result = ((x - min) / (max - min)) * (normalize - (-normalize)) - normalize
    return result


def getPcScale(point_cloud: np.ndarray) -> float:
    scale_x = np.max(point_cloud[:, 0]) - np.min(point_cloud[:, 0])
    scale_y = np.max(point_cloud[:, 1]) - np.min(point_cloud[:, 1])
    scale_z = np.max(point_cloud[:, 2]) - np.min(point_cloud[:, 2])
    return max(max(scale_x, scale_y), scale_z)


def centoroid(point, point_num):
    new_centor = []
    sum_x = sum(point[0]) / point_num
    sum_y = sum(point[1]) / point_num
    sum_z = sum(point[2]) / point_num
    centor_of_gravity = [sum_x, sum_y, sum_z]
    fractal_point_x = (point[0] - centor_of_gravity[0]).tolist()
    fractal_point_y = (point[1] - centor_of_gravity[1]).tolist()
    fractal_point_z = (point[2] - centor_of_gravity[2]).tolist()
    new_centor.append(fractal_point_x)
    new_centor.append(fractal_point_y)
    new_centor.append(fractal_point_z)
    new = np.array(new_centor)
    return new


def generate_point(params: np.ndarray, point_num: int, normalize: float = 1.0):
    fractal_point = generator(params, point_num)
    point = min_max(fractal_point, normalize, axis=None)
    point = centoroid(point, point_num)
    return point


class ifs_function:
    def __init__(self):
        self.prev_x, self.prev_y, self.prev_z = 0.0, 0.0, 0.0
        self.function = []
        self.xs, self.ys, self.zs = [], [], []
        self.select_function = []
        self.temp_proba = 0.0

    def set_param(self, a, b, c, d, e, f, g, h, i, j, k, l, proba, **kwargs):
        if "weight_a" in kwargs:
            a *= kwargs["weight_a"]
        if "weight_b" in kwargs:
            b *= kwargs["weight_b"]
        if "weight_c" in kwargs:
            c *= kwargs["weight_c"]
        if "weight_d" in kwargs:
            d *= kwargs["weight_d"]
        if "weight_e" in kwargs:
            e *= kwargs["weight_e"]
        if "weight_f" in kwargs:
            f *= kwargs["weight_f"]
        if "weight_g" in kwargs:
            g *= kwargs["weight_g"]
        if "weight_h" in kwargs:
            h *= kwargs["weight_h"]
        if "weight_i" in kwargs:
            i *= kwargs["weight_i"]
        if "weight_j" in kwargs:
            j *= kwargs["weight_j"]
        if "weight_k" in kwargs:
            k *= kwargs["weight_k"]
        if "weight_l" in kwargs:
            l *= kwargs["weight_l"]
        temp_function = {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e,
            "f": f,
            "g": g,
            "h": h,
            "i": i,
            "j": j,
            "k": k,
            "l": l,
            "proba": proba,
        }
        self.function.append(temp_function)
        self.temp_proba += proba
        self.select_function.append(self.temp_proba)

    def calculate(self, iteration):
        """Recursively calculate coordinates for args.iteration"""
        rand = np.random.random(iteration)
        select_function = self.select_function
        function = self.function
        prev_x, prev_y, prev_z = self.prev_x, self.prev_y, self.prev_z
        for i in range(iteration - 1):
            for j in range(len(select_function)):
                if rand[i] <= select_function[j]:
                    next_x = (
                        prev_x * function[j]["a"]
                        + prev_y * function[j]["b"]
                        + prev_z * function[j]["c"]
                        + function[j]["j"]
                    )
                    next_y = (
                        prev_x * function[j]["d"]
                        + prev_y * function[j]["e"]
                        + prev_z * function[j]["f"]
                        + function[j]["k"]
                    )
                    next_z = (
                        prev_x * function[j]["g"]
                        + prev_y * function[j]["h"]
                        + prev_z * function[j]["i"]
                        + function[j]["l"]
                    )
                    break
            self.xs.append(next_x), self.ys.append(next_y), self.zs.append(next_z)
            prev_x, prev_y, prev_z = next_x, next_y, next_z
        point_data = np.array((self.xs, self.ys, self.zs), dtype=float)
        return point_data
