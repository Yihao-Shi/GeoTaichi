#!/usr/bin/env python
import sys, os
try:
    import nose
except ImportError:
    nose = None

def main_legacy():
    import unittest
    from glob import glob
    testsuitedir = os.path.dirname(__file__)
    sys.path.insert(0, testsuitedir)
    pattern = 'test_*.py'
    wildcard = os.path.join(testsuitedir, pattern)
    testfiles = glob(wildcard)
    testfiles.sort()
    testsuite = unittest.TestSuite()
    testloader = unittest.TestLoader()
    for testfile in testfiles:
        filename = os.path.basename(testfile)
        testname = os.path.splitext(filename)[0]
        module = __import__(testname)
        cases = testloader.loadTestsFromModule(module)
        testsuite.addTests(cases)
        cases = []
        for attr in module.__dict__:
            if attr.startswith('test'):
                func = getattr(module, attr)
                case = unittest.FunctionTestCase(func)
                cases.append(case)
        testsuite.addTests(cases)
    runner = unittest.TextTestRunner()
    result = runner.run(testsuite)
    return result.wasSuccessful()

def main():
    if nose is None:
        return main_legacy()
    else:
        return nose.main()

if __name__ == '__main__':
    sys.dont_write_bytecode = True
    import bootstrap
    main()
