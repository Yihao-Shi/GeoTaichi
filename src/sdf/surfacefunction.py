from src.sdf.Surface import Surface
import types


class surfacefunction(Surface):
    def __init__(self, f=None, df=None, radial=None, bounding_box=None) -> None:
        super().__init__()
        self.fx = types.MethodType(f, self)
        self.dfx = types.MethodType(df, self)
        self.radial = types.MethodType(radial, self)
        self._estimate_bounding_box = types.MethodType(bounding_box, self)
