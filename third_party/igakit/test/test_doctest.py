import bootstrap
import doctest

def test_igakit_transform():
    from igakit import transform
    failures, _ = doctest.testmod(transform)
    assert failures==0
    
def test_igakit_nurbs():
    from igakit import nurbs
    failures, _ = doctest.testmod(nurbs)
    assert failures==0

def test_igakit_cad():
    from igakit import cad
    failures, _ = doctest.testmod(cad)
    assert failures==0

if __name__ == '__main__':
    test_igakit_transform()
    test_igakit_nurbs()
    test_igakit_cad()
