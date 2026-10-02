class IGAMaterialManager:
    def __init__(self):
        self.material = {}

    def add_material(self, **kwargs):
        self.material.update(kwargs)
        return self.material
