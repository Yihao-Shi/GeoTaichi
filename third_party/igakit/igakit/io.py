import numpy as np
from igakit.nurbs import NURBS

__all__ = ['PetIGA', 'VTK']

class PetIGA(object):

    """
    PetIGA_ Reader/Writer

    .. _PetIGA: https://github.com/dalcinl/petiga

    """

    VEC_ID = 1211214
    MAT_ID = 1211216
    IGA_ID = 1211299

    precision = {
        'single' : {'real' : '>f4', 'complex' : '>c8' },
        'double' : {'real' : '>f8', 'complex' : '>c16'},
        }
    indices = {
        '32bit' : '>i4',
        '64bit' : '>i8',
        }

    def __init__(self, precision='double',
                 scalar='real', indices='32bit'):
        I = self.indices[indices]
        R = self.precision[precision]['real']
        S = self.precision[precision][scalar]
        self._types = tuple(np.dtype(t) for t in (I, R, S))

    def _write(self, fid, dtype, array):
        np.asarray(array, dtype).tofile(fid)

    def _read(self, fid, dtype, count=None):
        if count is None:
            array = np.fromfile(fid, dtype, 1)[0]
        else:
            array = np.fromfile(fid, dtype, count)
        return array.astype(dtype.newbyteorder('='))

    def write(self, filename, nurbs,
              control=True, fields=True,
              nsd=None):
        """
        Parameters
        ----------
        filename : string
        nurbs : NURBS
        control : bool, optional
        fields : bool, optional
        nsd : int, optional

        """
        IGA_ID = self.IGA_ID
        VEC_ID = self.VEC_ID
        I,R,S  = self._types
        _write = self._write
        #
        info = 0
        dim = nurbs.dim
        knots = nurbs.knots
        degree = nurbs.degree
        Cw = None
        F  = None
        if control:
            if nsd is None: nsd = dim
            assert dim <= nsd <= 3
            idx = list(range(nsd))+[3]
            Cw = nurbs.control[...,idx]
            Cw = np.rollaxis(Cw, -1)
            info |= 0x1
        if fields:
            F = nurbs.fields
            if F is not None:
                F = np.rollaxis(F, -1)
                info |= 0x2
            else:
                fields = False
        #
        fh = open(filename, 'wb')
        try:
            _write(fh, I, IGA_ID)
            _write(fh, I, info)
            _write(fh, I, nurbs.dim)
            for p, U in zip(degree, knots):
                _write(fh, I, p)
                _write(fh, I, U.size)
                _write(fh, R, U)
            if control:
                _write(fh, I, Cw.shape[0]-1)
                _write(fh, I, VEC_ID)
                _write(fh, I, Cw.size)
                _write(fh, S, Cw.ravel('f'))
            if fields:
                _write(fh, I, F.shape[0])
                _write(fh, I, VEC_ID)
                _write(fh, I, F.size)
                _write(fh, S, F.ravel('f'))
        finally:
            fh.close()

    def read(self, filename):
        """
        Parameters
        ----------
        filename : string

        """
        IGA_ID = self.IGA_ID
        VEC_ID = self.VEC_ID
        I,R,S  = self._types
        _read  = self._read
        #
        fh = open(filename, 'rb')
        try:
            iga_id = _read(fh, I)
            assert iga_id == IGA_ID
            info = _read(fh, I)
            control = bool(info & 0x1)
            fields  = bool(info & 0x2)
            dim = _read(fh, I)
            assert 1 <= dim <= 3
            knots, degrs, sizes = [], [], []
            for i in range(dim):
                p = _read(fh, I)
                assert p >= 1
                m = _read(fh, I)
                n = m-p-1
                assert n >= 2
                U = _read(fh, R, m)
                assert len(U) == m
                degrs.append(p)
                sizes.append(n)
                knots.append(U)
            if control:
                nsd = _read(fh, I)
                assert dim <= nsd <= 3
                vec_id = _read(fh, I)
                assert vec_id == VEC_ID
                n  = _read(fh, I)
                Cw = _read(fh, S, n)
                assert len(Cw) == n
                if R != S:
                    Cw = np.real(Cw)
            else:
                Cw = None
            if fields:
                npd = _read(fh, I)
                assert npd >= 1
                vec_id = _read(fh, I)
                assert vec_id == VEC_ID
                n = _read(fh, I)
                D = _read(fh, S, n)
                assert len(D) == n
            else:
                D = None
        finally:
            fh.close()
        #
        if control:
            shape = [nsd+1] + sizes
            Cw = Cw.reshape(shape, order='f')
            Cw = np.rollaxis(Cw, 0, Cw.ndim)
            shape = sizes + [4]
            control = np.zeros(shape, dtype=Cw.dtype)
            control[..., :nsd] = Cw[..., :-1]
            control[...,   -1] = Cw[...,  -1]
        else:
            from igakit.igalib import bsp
            shape = sizes + [4]
            Cw = np.zeros(shape, dtype=S)
            for i in range(dim):
                X = bsp.Greville(degrs[i], knots[i])
                I = [np.newaxis] * dim
                I[i] = slice(None)
                I = tuple(I)
                Cw[...,i] = X[I]
            Cw[...,3] = 1
            control = Cw
        #
        if fields:
            shape = [npd] + sizes
            D = D.reshape(shape, order='f')
            D = np.rollaxis(D, 0, D.ndim)
            fields = D
        else:
            fields = None
        #
        return NURBS(knots, control, fields)

    def write_vec(self, filename, array, nurbs=None):
        VEC_ID = self.VEC_ID
        I,R,S  = self._types
        _write = self._write
        #
        A = np.asarray(array)
        if nurbs is not None:
            shape = nurbs.shape + (-1,)
            A = A.reshape(shape)
            A = np.rollaxis(A, -1)
            A = A.ravel('f')
        #
        fh = open(filename, 'wb')
        try:
            _write(fh, I, VEC_ID)
            _write(fh, I, A.size)
            _write(fh, S, A)
        finally:
            fh.close()

    def read_vec(self, filename, nurbs=None):
        VEC_ID = self.VEC_ID
        I,R,S  = self._types
        _read  = self._read
        #
        fh = open(filename, 'rb')
        try:
            clsid = _read(fh, I)
            assert clsid == VEC_ID
            n = _read(fh, I)
            A = _read(fh, S, n)
            assert len(A) == n
        finally:
            fh.close()
        #
        if nurbs is not None:
            shape = (-1,) + nurbs.shape
            A = A.reshape(shape, order='f')
            A = np.rollaxis(A, 0, A.ndim)
            A = A.squeeze()
            A = np.ascontiguousarray(A)
        return A

    def read_mat(self, filename):
        MAT_ID = self.MAT_ID
        I,R,S  = self._types
        _read  = self._read
        #
        fh = open(filename, 'rb')
        try:
            clsid = _read(fh, I)
            assert clsid == MAT_ID
            M, N, nz = _read(fh, I, 3)
            AI = np.empty(M+1, dtype=M.dtype)
            AI[0] = 0
            rownz = _read(fh, I, M)
            np.cumsum(rownz, out=AI[1:])
            assert AI[-1] == nz
            AJ = _read(fh, I, nz)
            assert len(AJ) == nz
            AV = np.fromfile(fh, S, nz)
            assert len(AV) == nz
        finally:
            fh.close()
        return (M, N), (AI, AJ, AV)


class VTK(object):

    """
    VTK_ Writer

    .. _VTK: http://www.vtk.org/

    """

    title = 'VTK Data'

    def __init__(self):
        pass

    def write(self, filename, nurbs,
              control=True, fields=None,
              scalars=(), vectors=(),
              sampler=None):
        """
        Parameters
        ----------
        filename : string
        nurbs : NURBS
        control : bool, optional
        fields : array, optional
        scalars : dict or sequence of 2-tuple, optional
        vectors : dict or sequence or 2-tuple, optional
        sampler : callable, optional

        """
        if sampler is None:
            sampler = lambda u: u
        dim  = nurbs.dim
        uvw = [sampler(u) for u in nurbs.breaks()]
        flag = bool(scalars or vectors)
        if not flag: fields = flag
        elif fields is None: fields = flag
        out = nurbs(*uvw, **dict(fields=fields))
        if flag: C, F = out
        else:    C, F = out, out[..., 0:0]

        dimensions = C.shape[:-1] + (1,)*(3-dim)
        coordinates = uvw + [np.zeros(1)]*(3-dim)
        points = np.rollaxis(C, -1).ravel('f')
        points.shape = (-1, 3)
        fields = np.rollaxis(F, -1).ravel('f')
        fields.shape = (len(points), -1)

        if isinstance(scalars, dict):
            keys = sorted(scalars.keys())
            scalars = [(k, scalars[k]) for k in keys]
        else:
            scalars = list(scalars)
        for i, (name, index) in enumerate(scalars):
            array = np.zeros((len(points), 1), dtype='d')
            array[:,0] = fields[:,index]
            scalars[i] = (name, array)

        if isinstance(vectors, dict):
            keys = sorted(vectors.keys())
            vectors = [(k, vectors[k]) for k in keys]
        else:
            vectors = list(vectors)
        for i, (name, index) in enumerate(vectors):
            array = np.zeros((len(points), 3), dtype='d')
            array[:,:len(index)] = fields[:,index]
            vectors[i] = (name, array)

        fh = open(filename, 'wb')
        fh_write = lambda s: fh.write(s.encode('ascii'))

        header = '# vtk DataFile Version %d.%d'
        version = (2, 0)
        fh_write(header % version)
        fh_write('\n')
        title = self.title
        fh_write(title[:255])
        fh_write('\n')

        format = 'BINARY'
        fh_write(format)
        fh_write('\n')

        if control:
            dataset_type = 'STRUCTURED_GRID'
            fh_write('DATASET %s' % dataset_type);
            fh_write('\n')
            fh_write('DIMENSIONS %d %d %d' % dimensions)
            fh_write('\n')
            fh_write('POINTS %d %s' % (len(points), 'double'))
            fh_write('\n')
            points.astype('>d').tofile(fh)
            fh_write('\n')
        else:
            dataset_type = 'RECTILINEAR_GRID'
            fh_write('DATASET %s' % dataset_type);
            fh_write('\n')
            fh_write('DIMENSIONS %d %d %d' % dimensions)
            fh_write('\n')
            for X, array in zip("XYZ", coordinates):
                label = X+'_COORDINATES'
                fh_write('%s %s %s' % (label, len(array), 'double'))
                fh_write('\n')
                array.astype('>d').tofile(fh)
                fh_write('\n')

        if (not scalars and
            not vectors):
            fh.flush()
            fh.close()
            return

        data_type = 'POINT_DATA'
        fh_write('%s %d' % (data_type, len(points)))
        fh_write('\n')

        for i, (name, array) in enumerate(scalars):
            attr_type = 'SCALARS'
            attr_name = name or (attr_type.lower() + str(i))
            attr_name = attr_name.replace(' ', '_')
            fh_write('%s %s %s' %(attr_type, attr_name, 'double'))
            fh_write('\n')
            lookup_table = 'default'
            lookup_table = lookup_table.replace(' ', '_')
            fh_write('LOOKUP_TABLE %s' % lookup_table)
            fh_write('\n')
            array.astype('>d').tofile(fh)
            fh_write('\n')

        for i, (name, array) in enumerate(vectors):
            attr_type = 'VECTORS'
            attr_name = name or (attr_type.lower() + str(i))
            attr_name = attr_name.replace(' ', '_')
            fh_write('%s %s %s' %(attr_type, attr_name, 'double'))
            fh_write('\n')
            array.astype('>d').tofile(fh)
            fh_write('\n')

        fh.flush()
        fh.close()
