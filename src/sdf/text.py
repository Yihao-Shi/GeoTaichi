from src.sdf.Utilities import _sdf

from PIL import Image, ImageFont, ImageDraw
import numpy as np
from src.sdf.SDF2D import SDF2D


class text(SDF2D):
    def __init__(self, texts: str, font_name="Arial", width=None, height=None, pixels=2**22, points=512):
        super().__init__()

        if width is None or height is None:
            width, height = self.measure_text(font_name, texts, height, width)

        font = ImageFont.truetype(font_name, points)
        # compute texture bounds
        p = 0.2
        x0, y0, x1, y1 = font.getbbox(texts)
        px = int((x1 - x0) * p)
        py = int((y1 - y0) * p)
        tw = x1 - x0 + 1 + px * 2
        th = y1 - y0 + 1 + py * 2

        # render text to image
        im = Image.new("L", (tw, th))
        draw = ImageDraw.Draw(im)
        draw.text((px - x0, py - y0), texts, font=font, fill=255)
        self.calculate_distance = _sdf(width, height, pixels, px, py, im)

    def measure_text(self, font_name, texts, width, height):
        font = ImageFont.truetype(font_name, 96)
        x0, y0, x1, y1 = font.getbbox(texts)
        aspect = (x1 - x0) / (y1 - y0)
        if width is None and height is None:
            height = 1
        if width is None:
            width = height * aspect
        if height is None:
            height = width / aspect
        return (width, height)


class image(SDF2D):
    def __init__(self, file: str, width=None, height=None, pixels=2**22):
        super().__init__()

        im = self._load_image(file)
        if width is None or height is None:
            width, height = self.measure_image(im, width, height)
        self.calculate_distance = _sdf(width, height, pixels, 0, 0, im)

    def _load_image(self, thing):
        if isinstance(thing, str):
            return Image.open(thing)
        elif isinstance(thing, (np.ndarray, np.generic)):
            return Image.fromarray(thing)
        return Image.fromarray(np.array(thing))

    def measure_image(self, im, width, height):
        w, h = im.size
        aspect = w / h
        if width is None and height is None:
            height = 1
        if width is None:
            width = height * aspect
        if height is None:
            height = width / aspect
        return width, height
