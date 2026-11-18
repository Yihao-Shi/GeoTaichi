import pytest

from src.utils.ObjectIO import DictIO


def test_normalized_properties_preserve_alias_order_and_false_values():
    properties = {"normal-stiffness": 0, "kn": 12, "FRICTION": None, "active": False}
    assert DictIO.get_normalized(properties, "NormalStiffness", "kn", required=True) == 0
    assert DictIO.get_normalized(properties, "friction", default=1) is None
    assert DictIO.get_normalized(properties, "ACTIVE", default=True) is False
    assert DictIO.get_normalized(properties, "missing", default=5) == 5
    assert DictIO.get_normalized({"kn": 12}, "NormalStiffness", "kn", required=True) == 12


def test_normalized_property_collisions_keep_last_value():
    assert DictIO.get_normalized({"Normal_Stiffness": 1, "NORMAL-STIFFNESS": 2}, "NormalStiffness") == 2


@pytest.mark.parametrize("context", ("FEDEM contact property", "FEMPM contact property"))
def test_normalized_required_property_retains_error_context(context):
    with pytest.raises(KeyError) as error:
        DictIO.get_normalized({}, "NormalStiffness", "kn", required=True, context=context)
    assert error.value.args == (f"{context} 'NormalStiffness' is required",)


def test_existing_dictionary_readers_retain_their_key_matching():
    assert DictIO.get_essential({"Normal_Stiffness": 4}, "NORMAL_STIFFNESS") == 4
    with pytest.raises(KeyError):
        DictIO.get_essential({"Normal_Stiffness": 4}, "NormalStiffness")
