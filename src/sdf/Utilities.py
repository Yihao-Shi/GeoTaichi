from __future__ import annotations
from src.utils.linalg import _bilinear_interpolate
import scipy.ndimage as nd
import numpy as np
from src.sdf.SDFs2D import rectangle


def _sdf(width, height, pixels, px, py, im):
    tw, th = im.size

    # downscale image if necessary
    factor = (pixels / (tw * th)) ** 0.5
    if factor < 1:
        tw, th = int(round(tw * factor)), int(round(th * factor))
        px, py = int(round(px * factor)), int(round(py * factor))
        im = im.resize((tw, th))

    # convert to numpy array and apply distance transform
    im = im.convert("1")
    a = np.array(im)
    inside = -nd.distance_transform_edt(a)
    outside = nd.distance_transform_edt(~a)
    texture = np.zeros(a.shape)
    texture[a] = inside[a]
    texture[~a] = outside[~a]

    # compute world bounds
    pw = tw - px * 2
    ph = th - py * 2
    aspect = pw / ph
    if width is None and height is None:
        height = 1
    if width is None:
        width = height * aspect
    if height is None:
        height = width / aspect
    x0 = -width / 2
    y0 = -height / 2
    x1 = width / 2
    y1 = height / 2

    # scale texture distances
    scale = width / tw
    texture *= scale

    # prepare fallback rectangle
    # TODO: reduce size based on mesh resolution instead of dividing by 2
    rect = rectangle(size=(width / 2, height / 2))

    def f(p):
        x = p[:, 0]
        y = p[:, 1]
        u = (x - x0) / (x1 - x0)
        v = (y - y0) / (y1 - y0)
        v = 1 - v
        i = u * pw + px
        j = v * ph + py
        d = _bilinear_interpolate(texture, i, j)
        q = rect(p).reshape(-1)
        outside = (i < 0) | (i >= tw - 1) | (j < 0) | (j >= th - 1)
        d[outside] = q[outside]
        return d

    return f
