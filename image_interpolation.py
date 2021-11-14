import numpy as np
import math
from time import time
from PIL.Image import open, fromarray


def resize_img(fp: str, scale, interpolator="bilinear", rotation=0, rotate_fit: bool = True):
    """Return a resized image by interporation.

    Parameters
    ----------
    fp : A file name (string).
    scale : float or tuple of float.
        A scale factor of image enlarging/reducing.
    interpolator : {"bilinear", "nearest"}, optional, default: "bilinear".
        An algorithm for interpolating images.
    rotation : float.
        How much of an angle to rotate the image.
    rotate_fit: bool.
        Whether or not to rotate the image without sticking out.

    """

    try:
        e = str(type(scale))[7:-1]
        if not isinstance(scale, tuple):
            scale = (scale, scale)
        scale = (float(scale[0]), float(scale[1]))
    except Exception:
        e = "scale " + e + " cannot be interpreted as an float or tuple of float."
        raise TypeError(e)

    itp = ["bilinear", "nearest"]

    if interpolator not in itp:
        if interpolator is not None:
            raise ValueError(f"invalid interpolator option: {interpolator}")
        else:
            raise TypeError("must specify interpolator")

    try:
        if rotation is None:
            rotation = 0
        e = str(type(rotation))[7:-1]
        rotation = float(rotation)
    except Exception:
        e = "rotation " + e + " cannot be interpreted as an float."
        raise TypeError(e)

    f = np.array(open(fp))
    tmp_shape = (round(f.shape[0] * scale[0]), round(f.shape[1] * scale[1]), f.shape[2])

    if rotation != 0:
        center_tmp = np.array(tmp_shape[:-1]) // 2
        corner = np.array([[0, 0] - center_tmp, [0, tmp_shape[1]] - center_tmp, [tmp_shape[0], 0] - center_tmp, [tmp_shape[0], tmp_shape[1]] - center_tmp])
        affine_rot = np.array([[math.cos(math.radians(rotation)), -math.sin(math.radians(rotation)), 0], [math.sin(math.radians(rotation)), math.cos(math.radians(rotation)), 0], [0, 0, 1]])
        inv_affine_rot = np.linalg.inv(affine_rot)

        if rotate_fit:
            corner_rot = map(lambda x: np.dot(affine_rot, x.T), np.insert(corner, 2, 0, axis=1))
            corner_rot = np.array(list(corner_rot))[:, :-1]
            x_max = round(corner_rot[:, 0].max())
            x_min = round(corner_rot[:, 0].min())
            y_max = round(corner_rot[:, 1].max())
            y_min = round(corner_rot[:, 1].min())
            g = np.zeros((x_max - x_min + 1, y_max - y_min + 1, f.shape[2]), dtype=np.uint8)
        else:
            g = np.zeros(tmp_shape, dtype=np.uint8)

        center_g = np.array(g.shape[:-1]) // 2
        affine_shift_g = np.array([[1, 0, -center_g[0]], [0, 1, -center_g[1]], [0, 0, 1]])
        affine_shift_back_tmp = np.array([[1, 0, center_tmp[0]], [0, 1, center_tmp[1]], [0, 0, 1]])
        affine_scale_back = np.array([[1 / scale[0], 0, 0], [0, 1 / scale[1], 0], [0, 0, 1]])
        affine_trans = np.dot(affine_scale_back, np.dot(affine_shift_back_tmp, np.dot(inv_affine_rot, affine_shift_g)))

    else:
        g = np.zeros(tmp_shape, dtype=np.uint8)

    print("Algorithm: " + interpolator)
    print("Scaling: " + str(f.shape) + " -> " + str(tmp_shape))
    if rotation != 0:
        print("Rotating:  0° -> {}°".format(str(rotation)))

    t_begin = time()

    for x in range(g.shape[0]):
        progress = x / (g.shape[0] - 1)
        bar = round(50 * progress)
        print("\r[{}] {: >3}%".format("#" * bar + "-" * (50 - bar), round(progress * 100)), end="")

        if rotation == 0:
            org_x = x / scale[0]
            if interpolator == "bilinear":
                ref_x = math.floor(org_x)
                s = org_x - ref_x

                if ref_x >= f.shape[0] - 1:
                    ref_x_next = ref_x
                else:
                    ref_x_next = ref_x + 1

            else:
                ref_x = round(org_x)

                if ref_x >= f.shape[0]:
                    ref_x = f.shape[0] - 1

        for y in range(g.shape[1]):
            if rotation != 0:
                org_x, org_y = np.dot(affine_trans, np.array([x, y, 1]))[:-1]
                if 0 <= org_x < f.shape[0] and 0 <= org_y < f.shape[1]:
                    if interpolator == "bilinear":
                        ref_x = math.floor(org_x)
                        ref_y = math.floor(org_y)
                        s = org_x - ref_x
                        t = org_y - ref_y

                        if ref_x >= f.shape[0] - 1:
                            ref_x_next = ref_x
                        else:
                            ref_x_next = ref_x + 1

                        if ref_y >= f.shape[1] - 1:
                            ref_y_next = ref_y
                        else:
                            ref_y_next = ref_y + 1

                        g[x, y] = (1 - s) * (1 - t) * f[ref_x, ref_y] + (1 - s) * t * f[ref_x, ref_y_next] + s * (1 - t) * f[ref_x_next, ref_y] + s * t * f[ref_x_next, ref_y_next]

                    else:
                        ref_x = round(org_x)
                        ref_y = round(org_y)

                        if ref_x >= f.shape[0]:
                            ref_x = f.shape[0] - 1
                        if ref_y >= f.shape[1]:
                            ref_y = f.shape[1] - 1

                        g[x, y] = f[ref_x, ref_y]

            else:
                org_y = y / scale[1]
                if interpolator == "bilinear":
                    ref_y = math.floor(org_y)
                    t = org_y - ref_y

                    if ref_y == f.shape[1] - 1:
                        ref_y_next = ref_y
                    else:
                        ref_y_next = ref_y + 1

                    g[x, y] = (1 - s) * (1 - t) * f[ref_x, ref_y] + (1 - s) * t * f[ref_x, ref_y_next] + s * (1 - t) * f[ref_x_next, ref_y] + s * t * f[ref_x_next, ref_y_next]

                else:
                    ref_y = round(org_y)

                    if ref_y >= f.shape[1]:
                        ref_y = f.shape[1] - 1

                    g[x, y] = f[ref_x, ref_y]

    print(" {:.3f}s".format(time() - t_begin))

    return g


def save(obj, fp: str):
    """Saves array-like object as an image.

    Parameters
    ----------
    obj: Array-like object.
    fp : A file name (string).

    """

    fromarray(obj).save(fp)
