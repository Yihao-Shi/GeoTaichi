import warnings, operator
import numpy as np
from keyword import iskeyword as _iskeyword
import sys as _sys

_ECONVERGED = 0
_ESIGNERR = -1
_ECONVERR = -2
_EVALUEERR = -3
_EINPROGRESS = 1

CONVERGED = 'converged'
SIGNERR = 'sign error'
CONVERR = 'convergence error'
VALUEERR = 'value error'
INPROGRESS = 'No error'

_tuplegetter = lambda index, doc: property(operator.itemgetter(index), doc=doc)

flag_map = {_ECONVERGED: CONVERGED, _ESIGNERR: SIGNERR, _ECONVERR: CONVERR,
            _EVALUEERR: VALUEERR, _EINPROGRESS: INPROGRESS}

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
            self.flag = 'unknown error %d' % (flag,)

    def __repr__(self):
        attrs = ['converged', 'flag', 'function_calls',
                 'iterations', 'root']
        m = max(map(len, attrs)) + 1
        return '\n'.join([a.rjust(m) + ': ' + repr(getattr(self, a))
                          for a in attrs])


def _results_select(full_output, r):
    """Select from a tuple of (root, funccalls, iterations, flag)"""
    x, funcalls, iterations, flag = r
    if full_output:
        results = RootResults(root=x,
                              iterations=iterations,
                              function_calls=funcalls,
                              flag=flag)
        return x, results
    return x


def newton(func, x0, fprime=None, args=(), tol=1.48e-8, maxiter=50,
           fprime2=None, x1=None, rtol=0.0,
           full_output=False, disp=True):
    if tol <= 0:
        raise ValueError("tol too small (%g <= 0)" % tol)
    maxiter = operator.index(maxiter)
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")
    if np.size(x0) > 1:
        return _array_newton(func, x0, fprime, args, tol, maxiter, fprime2,
                             full_output)

    # Convert to float (don't use float(x0); this works also for complex x0)
    p0 = 1.0 * x0
    funcalls = 0
    if fprime is not None:
        # Newton-Raphson method
        for itr in range(maxiter):
            # first evaluate fval
            fval = func(p0, *args)
            funcalls += 1
            # If fval is 0, a root has been found, then terminate
            if fval == 0:
                return _results_select(
                    full_output, (p0, funcalls, itr, _ECONVERGED))
            fder = fprime(p0, *args)
            funcalls += 1
            if fder == 0:
                msg = "Derivative was zero."
                if disp:
                    msg += (
                        " Failed to converge after %d iterations, value is %s."
                        % (itr + 1, p0))
                    raise RuntimeError(msg)
                warnings.warn(msg, RuntimeWarning)
                return _results_select(
                    full_output, (p0, funcalls, itr + 1, _ECONVERR))
            newton_step = fval / fder
            if fprime2:
                fder2 = fprime2(p0, *args)
                funcalls += 1
                # Halley's method:
                #   newton_step /= (1.0 - 0.5 * newton_step * fder2 / fder)
                # Only do it if denominator stays close enough to 1
                # Rationale: If 1-adj < 0, then Halley sends x in the
                # opposite direction to Newton. Doesn't happen if x is close
                # enough to root.
                adj = newton_step * fder2 / fder / 2
                if np.abs(adj) < 1:
                    newton_step /= 1.0 - adj
            p = p0 - newton_step
            if np.isclose(p, p0, rtol=rtol, atol=tol):
                return _results_select(
                    full_output, (p, funcalls, itr + 1, _ECONVERGED))
            p0 = p
    else:
        # Secant method
        if x1 is not None:
            if x1 == x0:
                raise ValueError("x1 and x0 must be different")
            p1 = x1
        else:
            eps = 1e-4
            p1 = x0 * (1 + eps)
            p1 += (eps if p1 >= 0 else -eps)
        q0 = func(p0, *args)
        funcalls += 1
        q1 = func(p1, *args)
        funcalls += 1
        if abs(q1) < abs(q0):
            p0, p1, q0, q1 = p1, p0, q1, q0
        for itr in range(maxiter):
            if q1 == q0:
                if p1 != p0:
                    msg = "Tolerance of %s reached." % (p1 - p0)
                    if disp:
                        msg += (
                            " Failed to converge after %d iterations, value is %s."
                            % (itr + 1, p1))
                        raise RuntimeError(msg)
                    warnings.warn(msg, RuntimeWarning)
                p = (p1 + p0) / 2.0
                return _results_select(
                    full_output, (p, funcalls, itr + 1, _ECONVERGED))
            else:
                if abs(q1) > abs(q0):
                    p = (-q0 / q1 * p1 + p0) / (1 - q0 / q1)
                else:
                    p = (-q1 / q0 * p0 + p1) / (1 - q1 / q0)
            if np.isclose(p, p1, rtol=rtol, atol=tol):
                return _results_select(
                    full_output, (p, funcalls, itr + 1, _ECONVERGED))
            p0, q0 = p1, q1
            p1 = p
            q1 = func(p1, *args)
            funcalls += 1

    if disp:
        msg = ("Failed to converge after %d iterations, value is %s."
               % (itr + 1, p))
        raise RuntimeError(msg)

    return _results_select(full_output, (p, funcalls, itr + 1, _ECONVERR))

def _array_newton(func, x0, fprime, args, tol, maxiter, fprime2, full_output):
    """
    A vectorized version of Newton, Halley, and secant methods for arrays.

    Do not use this method directly. This method is called from `newton`
    when ``np.size(x0) > 1`` is ``True``. For docstring, see `newton`.
    """
    # Explicitly copy `x0` as `p` will be modified inplace, but the
    # user's array should not be altered.
    p = np.array(x0, copy=True)

    failures = np.ones_like(p, dtype=bool)
    nz_der = np.ones_like(failures)
    if fprime is not None:
        # Newton-Raphson method
        for iteration in range(maxiter):
            # first evaluate fval
            fval = np.asarray(func(p, *args))
            # If all fval are 0, all roots have been found, then terminate
            if not fval.any():
                failures = fval.astype(bool)
                break
            fder = np.asarray(fprime(p, *args))
            nz_der = (fder != 0)
            # stop iterating if all derivatives are zero
            if not nz_der.any():
                break
            # Newton step
            dp = fval[nz_der] / fder[nz_der]
            if fprime2 is not None:
                fder2 = np.asarray(fprime2(p, *args))
                dp = dp / (1.0 - 0.5 * dp * fder2[nz_der] / fder[nz_der])
            # only update nonzero derivatives
            p = np.asarray(p, dtype=np.result_type(p, dp, np.float64))
            p[nz_der] -= dp
            failures[nz_der] = np.abs(dp) >= tol  # items not yet converged
            # stop iterating if there aren't any failures, not incl zero der
            if not failures[nz_der].any():
                break
    else:
        # Secant method
        dx = np.finfo(float).eps**0.33
        p1 = p * (1 + dx) + np.where(p >= 0, dx, -dx)
        q0 = np.asarray(func(p, *args))
        q1 = np.asarray(func(p1, *args))
        active = np.ones_like(p, dtype=bool)
        for iteration in range(maxiter):
            nz_der = (q1 != q0)
            # stop iterating if all derivatives are zero
            if not nz_der.any():
                p = (p1 + p) / 2.0
                break
            # Secant Step
            dp = (q1 * (p1 - p))[nz_der] / (q1 - q0)[nz_der]
            # only update nonzero derivatives
            p = np.asarray(p, dtype=np.result_type(p, p1, dp, np.float64))
            p[nz_der] = p1[nz_der] - dp
            active_zero_der = ~nz_der & active
            p[active_zero_der] = (p1 + p)[active_zero_der] / 2.0
            active &= nz_der  # don't assign zero derivatives again
            failures[nz_der] = np.abs(dp) >= tol  # not yet converged
            # stop iterating if there aren't any failures, not incl zero der
            if not failures[nz_der].any():
                break
            p1, p = p, p1
            q0 = q1
            q1 = np.asarray(func(p1, *args))

    zero_der = ~nz_der & failures  # don't include converged with zero-ders
    if zero_der.any():
        # Secant warnings
        if fprime is None:
            nonzero_dp = (p1 != p)
            # non-zero dp, but infinite newton step
            zero_der_nz_dp = (zero_der & nonzero_dp)
            if zero_der_nz_dp.any():
                rms = np.sqrt(
                    sum((p1[zero_der_nz_dp] - p[zero_der_nz_dp]) ** 2)
                )
                warnings.warn(
                    'RMS of {:g} reached'.format(rms), RuntimeWarning)
        # Newton or Halley warnings
        else:
            all_or_some = 'all' if zero_der.all() else 'some'
            msg = '{:s} derivatives were zero'.format(all_or_some)
            warnings.warn(msg, RuntimeWarning)
    elif failures.any():
        all_or_some = 'all' if failures.all() else 'some'
        msg = '{0:s} failed to converge after {1:d} iterations'.format(
            all_or_some, maxiter
        )
        if failures.all():
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning)

    if full_output:
        result = namedtuple('result', ('root', 'converged', 'zero_der'))
        p = result(p, ~failures, zero_der)

    return p

def namedtuple(typename, field_names, *, rename=False, defaults=None, module=None):
    """Returns a new subclass of tuple with named fields.

    >>> Point = namedtuple('Point', ['x', 'y'])
    >>> Point.__doc__                   # docstring for the new class
    'Point(x, y)'
    >>> p = Point(11, y=22)             # instantiate with positional args or keywords
    >>> p[0] + p[1]                     # indexable like a plain tuple
    33
    >>> x, y = p                        # unpack like a regular tuple
    >>> x, y
    (11, 22)
    >>> p.x + p.y                       # fields also accessible by name
    33
    >>> d = p._asdict()                 # convert to a dictionary
    >>> d['x']
    11
    >>> Point(**d)                      # convert from a dictionary
    Point(x=11, y=22)
    >>> p._replace(x=100)               # _replace() is like str.replace() but targets named fields
    Point(x=100, y=22)

    """

    # Validate the field names.  At the user's option, either generate an error
    # message or automatically replace the field name with a valid name.
    if isinstance(field_names, str):
        field_names = field_names.replace(',', ' ').split()
    field_names = list(map(str, field_names))
    typename = _sys.intern(str(typename))

    if rename:
        seen = set()
        for index, name in enumerate(field_names):
            if (not name.isidentifier()
                or _iskeyword(name)
                or name.startswith('_')
                or name in seen):
                field_names[index] = f'_{index}'
            seen.add(name)

    for name in [typename] + field_names:
        if type(name) is not str:
            raise TypeError('Type names and field names must be strings')
        if not name.isidentifier():
            raise ValueError('Type names and field names must be valid '
                             f'identifiers: {name!r}')
        if _iskeyword(name):
            raise ValueError('Type names and field names cannot be a '
                             f'keyword: {name!r}')

    seen = set()
    for name in field_names:
        if name.startswith('_') and not rename:
            raise ValueError('Field names cannot start with an underscore: '
                             f'{name!r}')
        if name in seen:
            raise ValueError(f'Encountered duplicate field name: {name!r}')
        seen.add(name)

    field_defaults = {}
    if defaults is not None:
        defaults = tuple(defaults)
        if len(defaults) > len(field_names):
            raise TypeError('Got more default values than field names')
        field_defaults = dict(reversed(list(zip(reversed(field_names),
                                                reversed(defaults)))))

    # Variables used in the methods and docstrings
    field_names = tuple(map(_sys.intern, field_names))
    num_fields = len(field_names)
    arg_list = repr(field_names).replace("'", "")[1:-1]
    repr_fmt = '(' + ', '.join(f'{name}=%r' for name in field_names) + ')'
    tuple_new = tuple.__new__
    _dict, _tuple, _len, _map, _zip = dict, tuple, len, map, zip

    # Create all the named tuple methods to be added to the class namespace

    s = f'def __new__(_cls, {arg_list}): return _tuple_new(_cls, ({arg_list}))'
    namespace = {'_tuple_new': tuple_new, '__name__': f'namedtuple_{typename}'}
    # Note: exec() has the side-effect of interning the field names
    exec(s, namespace)
    __new__ = namespace['__new__']
    __new__.__doc__ = f'Create new instance of {typename}({arg_list})'
    if defaults is not None:
        __new__.__defaults__ = defaults

    @classmethod
    def _make(cls, iterable):
        result = tuple_new(cls, iterable)
        if _len(result) != num_fields:
            raise TypeError(f'Expected {num_fields} arguments, got {len(result)}')
        return result

    _make.__func__.__doc__ = (f'Make a new {typename} object from a sequence '
                              'or iterable')

    def _replace(self, /, **kwds):
        result = self._make(_map(kwds.pop, field_names, self))
        if kwds:
            raise ValueError(f'Got unexpected field names: {list(kwds)!r}')
        return result

    _replace.__doc__ = (f'Return a new {typename} object replacing specified '
                        'fields with new values')

    def __repr__(self):
        'Return a nicely formatted representation string'
        return self.__class__.__name__ + repr_fmt % self

    def _asdict(self):
        'Return a new dict which maps field names to their values.'
        return _dict(_zip(self._fields, self))

    def __getnewargs__(self):
        'Return self as a plain tuple.  Used by copy and pickle.'
        return _tuple(self)

    # Modify function metadata to help with introspection and debugging
    for method in (__new__, _make.__func__, _replace,
                   __repr__, _asdict, __getnewargs__):
        method.__qualname__ = f'{typename}.{method.__name__}'

    # Build-up the class namespace dictionary
    # and use type() to build the result class
    class_namespace = {
        '__doc__': f'{typename}({arg_list})',
        '__slots__': (),
        '_fields': field_names,
        '_field_defaults': field_defaults,
        # alternate spelling for backward compatibility
        '_fields_defaults': field_defaults,
        '__new__': __new__,
        '_make': _make,
        '_replace': _replace,
        '__repr__': __repr__,
        '_asdict': _asdict,
        '__getnewargs__': __getnewargs__,
    }
    for index, name in enumerate(field_names):
        doc = _sys.intern(f'Alias for field number {index}')
        class_namespace[name] = _tuplegetter(index, doc)

    result = type(typename, (tuple,), class_namespace)

    # For pickling to work, the __module__ variable needs to be set to the frame
    # where the named tuple is created.  Bypass this step in environments where
    # sys._getframe is not defined (Jython for example) or sys._getframe is not
    # defined for arguments greater than 0 (IronPython), or where the user has
    # specified a particular module.
    if module is None:
        try:
            module = _sys._getframe(1).f_globals.get('__name__', '__main__')
        except (AttributeError, ValueError):
            pass
    if module is not None:
        result.__module__ = module

    return result


import taichi as ti
@ti.func
def cubic_roots(coef, x0, x1, tol=1e-10):
    """
    implemets cubic roots as https://github.com/cemyuksel/cyCodeBase/blob/master/cyPolynomial.h
    Finds the roots of the cubic polynomial between x0 and x1 with tol and returns the roots.
    :param coef: vector([d,c,b,a]) f = a * x*x*x + b * x*x + c * x + d
    :param x0: x_min
    :param x1: x_max
    :param tol:
    :return: vector([root0,root1,root2]) if there are less than 3 roots, return 10
    """
    ret = False
    roots_0 = 10.0
    roots_1 = 10.0
    roots_2 = 10.0
    y0 = cubic_eval(coef, x0)
    y1 = cubic_eval(coef, x1)
    a = coef[3] *3
    b_2 = coef[2]
    c = coef[1]
    deriv = ti.Vector([c, 2 * b_2, a, 0])
    delta_4 = b_2 * b_2 - a * c
    if delta_4 > 0:
        d_2 = ti.sqrt(delta_4)
        q = - (b_2 + d_2 * NewSign(b_2))
        rv0 = q / a
        rv1 = c / q
        xa = ti.min(rv0, rv1)
        xb = ti.max(rv0, rv1)
        if IsDifferentSign(y0,y1):
            if xa >= x1 or xb <= x0 or (xa <= x0 and xb >= x1):
                roots_0 = FindClosed(coef, deriv, x0, x1, y0, tol)
                ret = True
        else:
            if (xa >= x1 or xb <= x0) or (xa <= x0 and xb >= x1):
                ret = True

        if ret == False:
            if xa > x0:
                ya = cubic_eval(coef, xa)
                if IsDifferentSign(y0,ya):
                    roots_0 = FindClosed(coef, deriv, x0, xa, y0, tol)
                    if IsDifferentSign(ya,y1) or (xb < x1 and IsDifferentSign(ya, cubic_eval(coef, xb))):
                        defPoly0,defPoly1,defPoly2 = PolynomialDeflate(coef, roots_0)
                        roots_1, roots_2 = QuadraticRoots(defPoly0, defPoly1, defPoly2, xa, x1)
                elif xb < x1:
                    yb = cubic_eval(coef, xb)
                    if IsDifferentSign(ya,yb):
                        roots_0 = FindClosed(coef, deriv, xa, xb, ya, tol)
                        if IsDifferentSign(yb,y1):
                            defPoly0, defPoly1, defPoly2 = PolynomialDeflate(coef, roots_0)
                            roots_1, roots_2 = QuadraticRoots(defPoly0, defPoly1, defPoly2, xb, x1)
                    elif IsDifferentSign(yb,y1):
                        roots_0 = FindClosed(coef, deriv, xb, x1, yb, tol)
                elif IsDifferentSign(ya,y1):
                    roots_0 = FindClosed(coef, deriv, xa, x1, ya, tol)
            else:
                yb = cubic_eval(coef, xb)
                if IsDifferentSign(y0,yb):
                    roots_0 = FindClosed(coef, deriv, x0, xb,  y0, tol)
                    if IsDifferentSign(yb,y1):
                        defPoly0, defPoly1, defPoly2 = PolynomialDeflate(coef, roots_0)
                        roots_1, roots_2 = QuadraticRoots(defPoly0, defPoly1, defPoly2, xb, x1)
                elif IsDifferentSign(yb,y1):
                    roots_0 = FindClosed(coef, deriv, xb, x1, yb, tol)
    elif IsDifferentSign(y0,y1):
        roots_0 = FindClosed(coef, deriv, x0,  x1, y0, tol)
    return roots_0,roots_1,roots_2

@ti.func
def cubic_first_root(coef, x0, x1, tol=6e-4):
    """
    Finds the first root of the cubic polynomial between x0 and x1 with tol and returns the root.
    :param coef: vector([d,c,b,a]) f = a * x*x*x + b * x*x + c * x + d
    !!check if it returns the minimal root
    :param x0: x_min
    :param x1: x_max
    :param tol:
    :return: float
    """
    ret = False
    root = 10.0
    y0 = cubic_eval(coef, x0)
    y1 = cubic_eval(coef, x1)
    a = coef[3] * 3
    b_2 = coef[2]
    c = coef[1]
    deriv = ti.Vector([c, 2*b_2, a, 0])
    delta_4 = b_2*b_2 - a*c
    if delta_4 > 0:
        d_2 = ti.sqrt(delta_4)
        q = - ( b_2 + d_2 * NewSign(b_2) )
        rv0 = q / a
        rv1 = c / q
        xa = ti.min(rv0, rv1)
        xb = ti.max(rv0, rv1)
        if IsDifferentSign(y0,y1):
            if xa >= x1 or xb <= x0 or ( xa <= x0 and xb >= x1 ):
                root = FindClosed(coef, deriv, x0, x1, y0, tol)
                ret = True
        else:
            if (xa >= x1 or xb <= x0) or ( xa <= x0 and xb >= x1 ):
                ret = True

        if ret == False:
            if xa > x0:
                ya = cubic_eval(coef, xa)
                if IsDifferentSign(y0,ya):
                    root = FindClosed(coef, deriv, x0, xa, y0, tol)
                elif xb < x1:
                    yb = cubic_eval(coef, xb)
                    if IsDifferentSign(ya,yb):
                        root = FindClosed(coef, deriv, xa, xb, ya, tol)
                    elif IsDifferentSign(yb,y1):
                        root = FindClosed(coef, deriv, xb, x1, yb, tol)
                elif IsDifferentSign(ya,y1):
                    root = FindClosed(coef, deriv, xa, x1, ya, tol)
            else:
                yb = cubic_eval(coef, xb)
                if IsDifferentSign(y0,yb):
                    root = FindClosed(coef, deriv, x0, xb,  y0, tol)
                elif IsDifferentSign(yb,y1):
                    root = FindClosed(coef, deriv, xb, x1, yb, tol)
    elif IsDifferentSign(y0,y1):
        root = FindClosed(coef, deriv, x0,  x1, y0, tol)
    return root

@ti.func
def CubicHasRoot(coef, x0, x1):
    ret = 0
    y0 = cubic_eval(coef, x0)
    y1 = cubic_eval(coef, x1)
    if IsDifferentSign(y0,y1):
        ret = 1
    else:
        a = coef[3] * 3.
        b_2 = coef[2]
        c = coef[1]
        delta_4 = b_2 * b_2 - a * c
        if delta_4 > 0:
            d_2 = ti.sqrt(delta_4)
            q = - (b_2 + d_2 * NewSign(b_2))
            rv0 = q / a
            rv1 = c / q
            xa = ti.min(rv0, rv1)
            xb = ti.max(rv0, rv1)
            if (xa >= x1 or xb <= x0) or ( xa <= x0 and xb >= x1 ):
                ret = 0
            elif xa > x0:
                ya = cubic_eval(coef, xa)
                if IsDifferentSign(y0,ya):
                    ret = 1
                elif xb < x1:
                    yb = cubic_eval(coef, xb)
                    if IsDifferentSign(y0,yb):
                        ret = 1
            elif xa <= x0:
                yb = cubic_eval(coef, xb)
                if IsDifferentSign(y0,yb):
                    ret = 1
    return ret

@ti.func
def QuadraticRoots(defPoly0, defPoly1, defPoly2, x0, x1):
    roots_0 = 10.0
    roots_1 = 10.0
    c = defPoly0
    b = defPoly1
    a = defPoly2
    delta = b * b - 4 * a * c
    if delta > 0:
        d = ti.sqrt(delta)
        q = -0.5 * (b + d * NewSign(b))
        rv0 = q / a
        rv1 = c / q
        r0 = ti.min(rv0, rv1)
        r1 = ti.max(rv0, rv1)
        if (r0 >= x0) and (r0 <= x1):
            roots_0 = r0
        if (r1 >= x0) and (r1 <= x1):
            roots_1 = r1
    elif delta == 0:
        r0 = -0.5 * b / a
        if (r0 >= x0) and (r0 <= x1):
            roots_0 = r0
    return roots_0, roots_1

@ti.func
def PolynomialDeflate(coef, root):
    defPoly2 = coef[3]
    defPoly1 = coef[2] + root * defPoly2
    defPoly0 = coef[1] + root * defPoly1
    return defPoly0, defPoly1, defPoly2

@ti.func
def cubic_eval(coef, x):
    return x * (x * (coef[3] * x + coef[2]) + coef[1]) + coef[0]

@ti.func
def NewSign(x):
    return ti.cast((x >= 0) - (x < 0), float)

@ti.func
def IsDifferentSign(a, b):
    return (a < 0) != (b < 0)

@ti.func
def FindClosed(coef, deriv, x0, x1, y0, xError):
    ep2 = 2 * xError
    xr = (x0 + x1) / 2
    ret = False
    if x1 - x0 > ep2:
        xr0 = xr
        for _ in range(16):
            xn = xr - cubic_eval(coef, xr) / cubic_eval(deriv, xr)
            xn = ti.max(x0, ti.min(x1, xn))
            if abs(xr - xn) <= xError:
                ret = True
                xr = xn
                break
            xr = xn
        if ret == False:
            if not ti.math.isinf(xr):
                xr = xr0
            yr = cubic_eval(coef, xr)
            xb0 = x0
            xb1 = x1
            while True:
                side = IsDifferentSign(y0,yr)
                if side:
                    xb1 = xr
                else:
                    xb0 = xr
                dy = cubic_eval(deriv, xr)
                dx = yr / dy
                xn = xr - dx
                if (xn > xb0) and (xn < xb1):
                    stepsize = ti.abs(xr - xn)
                    xr = xn
                    if stepsize > xError:
                        yr = cubic_eval(coef, xr)
                    else:
                        break
                else:
                    xr = (xb0 + xb1) / 2
                    if (xr == xb0) or (xr == xb1) or (xb1 - xb0 <= ep2):
                        break
                    yr = cubic_eval(coef, xr)
    return xr