DIM = 2
DYNAMIC = True


def set_dimension(dimension: int):
    global DIM
    if int(dimension) not in (2, 3):
        raise ValueError("IGA-MPM coupling dimension must be 2 or 3")
    DIM = int(dimension)

    import src.iga.config as iga_config
    import src.mpm.config as mpm_config

    iga_config.set_dimension(DIM)
    mpm_config.set_dimension(DIM)


def get_dimension():
    return int(DIM)


def normalize_contact_model(contact_model):
    if contact_model not in ("BarrierIPC", "Linear", "HertzMindlin"):
        raise ValueError("IGA-MPM contact_model must be BarrierIPC, Linear, or HertzMindlin")
    return contact_model
