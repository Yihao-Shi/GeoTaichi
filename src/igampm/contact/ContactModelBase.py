class ContactModelBase:
    def __init__(self, max_material_num=1):
        self.max_material_num = int(max_material_num)
        self.surfaceProps = None
        self.null_model = True
        self.model_type = -1

    def get_componousID(self, materialID1, materialID2):
        return int(materialID1 * self.max_material_num + materialID2)

    def add_surface_property(self, materialID1, materialID2, property):
        raise NotImplementedError

    def update_property(self, materialID1, materialID2, property_name, value, override=True):
        raise NotImplementedError
