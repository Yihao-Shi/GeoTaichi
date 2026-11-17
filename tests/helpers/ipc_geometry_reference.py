"""Host feature-numbering oracle for tests only."""


def legacy_ee_type_to_ipc_py(feature_type):
    """Host equivalent of :func:`legacy_ee_type_to_ipc`."""
    mapping = (0, 1, 6, 2, 3, 7, 4, 5, 8)
    feature_type = int(feature_type)
    if feature_type < 0 or feature_type >= len(mapping):
        raise ValueError("legacy EE feature type must be in [0, 8]")
    return mapping[feature_type]


def ipc_ee_type_to_legacy_py(feature_type):
    """Host equivalent of :func:`ipc_ee_type_to_legacy`."""
    mapping = (0, 1, 3, 4, 6, 7, 2, 5, 8)
    feature_type = int(feature_type)
    if feature_type < 0 or feature_type >= len(mapping):
        raise ValueError("IPC EE feature type must be in [0, 8]")
    return mapping[feature_type]
