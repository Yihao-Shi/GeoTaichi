_ECONVERGED = 0
_ESIGNERR = -1
_ECONVERR = -2
_EVALUEERR = -3
_EINPROGRESS = 1
CONVERGED = "converged"
SIGNERR = "sign error"
CONVERR = "convergence error"
VALUEERR = "value error"
INPROGRESS = "No error"
flag_map = {
    _ECONVERGED: CONVERGED,
    _ESIGNERR: SIGNERR,
    _ECONVERR: CONVERR,
    _EVALUEERR: VALUEERR,
    _EINPROGRESS: INPROGRESS,
}


class RootResults:
    """Represents the root finding result.

    Attributes
    ----------
    root : float
        Estimated root location.
    iterations : int
        Number of iterations needed to find the root.
    function_calls : int
        Number of times the function was called.
    converged : bool
        True if the routine converged.
    flag : str
        Description of the cause of termination.

    """

    def __init__(self, root, iterations, function_calls, flag):
        self.root = root
        self.iterations = iterations
        self.function_calls = function_calls
        self.converged = flag == _ECONVERGED
        self.flag = None
        try:
            self.flag = flag_map[flag]
        except KeyError:
            self.flag = "unknown error %d" % (flag,)

    def __repr__(self):
        attrs = ["converged", "flag", "function_calls", "iterations", "root"]
        m = max(map(len, attrs)) + 1
        return "\n".join([a.rjust(m) + ": " + repr(getattr(self, a)) for a in attrs])
