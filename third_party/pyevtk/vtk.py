# ***********************************************************************************
# * Copyright 2010 - 2016 Paulo A. Herrera. All rights reserved.                    *
# *                                                                                 *
# * Redistribution and use in source and binary forms, with or without              *
# * modification, are permitted provided that the following conditions are met:     *
# *                                                                                 *
# *  1. Redistributions of source code must retain the above copyright notice,      *
# *  this list of conditions and the following disclaimer.                          *
# *                                                                                 *
# *  2. Redistributions in binary form must reproduce the above copyright notice,   *
# *  this list of conditions and the following disclaimer in the documentation      *
# *  and/or other materials provided with the distribution.                         *
# *                                                                                 *
# * THIS SOFTWARE IS PROVIDED BY PAULO A. HERRERA ``AS IS'' AND ANY EXPRESS OR      *
# * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF    *
# * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO      *
# * EVENT SHALL <COPYRIGHT HOLDER> OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,        *
# * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,  *
# * BUT NOT LIMITED TO, PROCUREMEN OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    *
# * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY           *
# * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING  *
# * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS              *
# * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                    *
# ***********************************************************************************
"""Low level Python library to export data to binary VTK file."""

import sys
import os
import numpy as np

from .evtk import writeBlockSize, writeArrayToFile, writeArraysToFile
from .xml import XmlWriter


# ================================
#            VTK Types
# ================================

#     FILE TYPES
class VtkFileType:
    """
    Wrapper class for VTK file types.

    Parameters
    ----------
    name : str
        Data name.
    ext : str
        File extension.
    """

    def __init__(self, name, ext):
        self.name = name
        self.ext = ext

    def __str__(self):
        return "Name: %s  Ext: %s \n" % (self.name, self.ext)


VtkImageData = VtkFileType("ImageData", ".vti")
VtkPolyData = VtkFileType("PolyData", ".vtp")
VtkRectilinearGrid = VtkFileType("RectilinearGrid", ".vtr")
VtkStructuredGrid = VtkFileType("StructuredGrid", ".vts")
VtkUnstructuredGrid = VtkFileType("UnstructuredGrid", ".vtu")


class VtkParallelFileType:
    """
    A wrapper class for parallel vtk file types.

    Parameters
    ----------
    vtkftype : VtkFileType
        Vtk file type
    """

    def __init__(self, vtkftype):
        self.name = "P" + vtkftype.name
        ext = vtkftype.ext
        self.ext = ext[0] + "p" + ext[1:]


VtkPImageData = VtkParallelFileType(VtkImageData)
VtkPPolyData = VtkParallelFileType(VtkPolyData)
VtkPRectilinearGrid = VtkParallelFileType(VtkRectilinearGrid)
VtkPStructuredGrid = VtkParallelFileType(VtkStructuredGrid)
VtkPUnstructuredGrid = VtkParallelFileType(VtkUnstructuredGrid)


#    DATA TYPES
class VtkDataType:
    """
    Wrapper class for VTK data types.

    Parameters
    ----------
    size : int
        Size in byte.
    name : str
        Type name.
    """

    def __init__(self, size, name):
        self.size = size
        self.name = name

    def __str__(self):
        return "Type: %s  Size: %d \n" % (self.name, self.size)


VtkInt8 = VtkDataType(1, "Int8")
VtkUInt8 = VtkDataType(1, "UInt8")
VtkInt16 = VtkDataType(2, "Int16")
VtkUInt16 = VtkDataType(2, "UInt16")
VtkInt32 = VtkDataType(4, "Int32")
VtkUInt32 = VtkDataType(4, "UInt32")
VtkInt64 = VtkDataType(8, "Int64")
VtkUInt64 = VtkDataType(8, "UInt64")
VtkFloat32 = VtkDataType(4, "Float32")
VtkFloat64 = VtkDataType(8, "Float64")

# Map numpy to VTK data types
np_to_vtk = {
    "int8": VtkInt8,
    "uint8": VtkUInt8,
    "int16": VtkInt16,
    "uint16": VtkUInt16,
    "int32": VtkInt32,
    "uint32": VtkUInt32,
    "int64": VtkInt64,
    "uint64": VtkUInt64,
    "float32": VtkFloat32,
    "float64": VtkFloat64,
}


#    CELL TYPES
class VtkCellType:
    """
    Wrapper class for VTK cell types.

    Parameters
    ----------
    tid : int
        Type ID.
    name : str
        Cell type name.
    """

    def __init__(self, tid, name):
        self.tid = tid
        self.name = name

    def __str__(self):
        return "VtkCellType( %s ) \n" % (self.name)


VtkVertex = VtkCellType(1, "Vertex")
VtkPolyVertex = VtkCellType(2, "PolyVertex")
VtkLine = VtkCellType(3, "Line")
VtkPolyLine = VtkCellType(4, "PolyLine")
VtkTriangle = VtkCellType(5, "Triangle")
VtkTriangleStrip = VtkCellType(6, "TriangleStrip")
VtkPolygon = VtkCellType(7, "Polygon")
VtkPixel = VtkCellType(8, "Pixel")
VtkQuad = VtkCellType(9, "Quad")
VtkTetra = VtkCellType(10, "Tetra")
VtkVoxel = VtkCellType(11, "Voxel")
VtkHexahedron = VtkCellType(12, "Hexahedron")
VtkWedge = VtkCellType(13, "Wedge")
VtkPyramid = VtkCellType(14, "Pyramid")
VtkQuadraticEdge = VtkCellType(21, "Quadratic_Edge")
VtkQuadraticTriangle = VtkCellType(22, "Quadratic_Triangle")
VtkQuadraticQuad = VtkCellType(23, "Quadratic_Quad")
VtkQuadraticTetra = VtkCellType(24, "Quadratic_Tetra")
VtkQuadraticHexahedron = VtkCellType(25, "Quadratic_Hexahedron")


# ==============================
#       Helper functions
# ==============================
def _mix_extents(start, end):
    assert len(start) == len(end) == 3
    string = "%d %d %d %d %d %d" % (
        start[0],
        end[0],
        start[1],
        end[1],
        start[2],
        end[2],
    )
    return string


def _array_to_string(a):
    s = "".join([repr(num) + " " for num in a])
    return s


def _get_byte_order():
    if sys.byteorder == "little":
        return "LittleEndian"
    return "BigEndian"


# ================================
#        VtkGroup class
# ================================
class VtkGroup:
    """
    Creates a VtkGroup file that is stored in filepath.

    Parameters
    ----------
    filepath : str
        filename without extension.
    """

    def __init__(self, filepath):
        self.xml = XmlWriter(filepath + ".pvd")
        self.xml.openElement("VTKFile")
        self.xml.addAttributes(
            type="Collection", version="0.1", byte_order=_get_byte_order()
        )
        self.xml.openElement("Collection")
        self.root = os.path.dirname(filepath)

    def save(self):
        """Close this VtkGroup."""
        self.xml.closeElement("Collection")
        self.xml.closeElement("VTKFile")
        self.xml.close()

    def addFile(self, filepath, sim_time, group="", part="0", name=""):
        """
        Add a file to this VTK group.

        Parameters
        ----------
        filepath : str
            full path to VTK file.
        sim_time : float
            simulated time.
        group : str, optional
            This attribute is not required;
            it is only for informational purposes.
            The default is "".
        part : int, optional
            It is an integer value greater than or equal to 0.
            The default is "0".

        Notes
        -----
        See: http://www.paraview.org/Wiki/ParaView/Data_formats#PVD_File_Format for details.
        """
        # TODO: Check what the other attributes are for.
        filename = os.path.relpath(filepath, start=self.root)
        self.xml.openElement("DataSet")
        self.xml.addAttributes(
            timestep=sim_time, group=group, part=part, file=filename, name=name
        )
        self.xml.closeElement()


# ================================
#        VtkFile class
# ================================
class VtkFile:
    """
    Class for a VTK file.

    Parameters
    ----------
    filepath : str
        filename without extension.
    ftype : str
        file type, e.g. VtkImageData, etc.
    largeFile : bool, optional
        If size of the stored data cannot be represented by a UInt32.
        The default is False.
    """

    def __init__(self, filepath, ftype):
        self.ftype = ftype
        self.filename = filepath + ftype.ext
        self.xml = XmlWriter(self.filename)
        self.offset = 0  # offset in bytes after beginning of binary section
        self.appendedDataIsOpen = False
        self.xml.openElement("VTKFile").addAttributes(
            type=ftype.name,
            version="1.0",
            byte_order=_get_byte_order(),
            header_type="UInt64",
        )

    def getFileName(self):
        """Return absolute path to this file."""
        return os.path.abspath(self.filename)

    def openPiece(
        self,
        start=None,
        end=None,
        npoints=None,
        ncells=None,
        nverts=None,
        nlines=None,
        nstrips=None,
        npolys=None,
    ):
        """
        Open piece section.

        Parameters
        ----------
        start : array-like, optional
            array or list with start indexes in each direction.
            Must be given with end.
        end : array-like, optional
            array or list with end indexes in each direction.
            Must be given with start.
        npoints : int, optional
            number of points in piece
        ncells : int, optional
            number of cells in piece.
            If present, npoints must also be given.
        nverts : int, optional
            number of vertices.
        nlines : int, optional
            number of lines.
        nstrips : int, optional
            number of stripes.
        npolys : int, optional
            number of poly.

        Returns
        -------
        VtkFile
            This VtkFile to allow chained calls.
        """
        # TODO: Check what are the requirements for each type of grid.

        self.xml.openElement("Piece")
        if start and end:
            ext = _mix_extents(start, end)
            self.xml.addAttributes(Extent=ext)

        elif ncells and npoints:
            self.xml.addAttributes(NumberOfPoints=npoints, NumberOfCells=ncells)

        elif npoints or nverts or nlines or nstrips or npolys:
            if npoints is None:
                npoints = str(0)
            if nverts is None:
                nverts = str(0)
            if nlines is None:
                nlines = str(0)
            if nstrips is None:
                nstrips = str(0)
            if npolys is None:
                npolys = str(0)
            self.xml.addAttributes(
                NumberOfPoints=npoints,
                NumberOfVerts=nverts,
                NumberOfLines=nlines,
                NumberOfStrips=nstrips,
                NumberOfPolys=npolys,
            )
        else:
            assert False

        return self

    def closePiece(self):
        """Close Piece."""
        self.xml.closeElement("Piece")

    def openData(
        self,
        nodeType,
        scalars=None,
        vectors=None,
        normals=None,
        tensors=None,
        tcoords=None,
    ):
        """
        Open data section.

        Parameters
        ----------
        nodeType : str
            Either "Point", "Cell" or "Field".
        scalars : str, optional
            default data array name for scalar data.
        vectors : str, optional
            default data array name for vector data.
        normals : str, optional
            default data array name for normals data.
        tensors : str, optional
            default data array name for tensors data.
        tcoords : str, optional
            default data array name for tcoords data.

        Returns
        -------
        VtkFile
            This VtkFile to allow chained calls.
        """
        self.xml.openElement(nodeType + "Data")
        if scalars:
            self.xml.addAttributes(Scalars=scalars)
        if vectors:
            self.xml.addAttributes(Vectors=vectors)
        if normals:
            self.xml.addAttributes(Normals=normals)
        if tensors:
            self.xml.addAttributes(Tensors=tensors)
        if tcoords:
            self.xml.addAttributes(TCoords=tcoords)

        return self

    def closeData(self, nodeType):
        """
        Close data section.

        Parameters
        ----------
        nodeType : str
            "Point", "Cell" or "Field".

        Returns
        -------
        VtkFile
            This VtkFile to allow chained calls.
        """
        self.xml.closeElement(nodeType + "Data")

    def openGrid(self, start=None, end=None, origin=None, spacing=None):
        """
        Open grid section.

        Parameters
        ----------
        start : array-like, optional
            array or list of start indexes.
            Required for Structured, Rectilinear and ImageData grids.
            The default is None.
        end : array-like, optional
            array or list of end indexes.
            Required for Structured, Rectilinear and ImageData grids.
            The default is None.
        origin : array-like, optional
            3D array or list with grid origin.
            Only required for ImageData grids.
            The default is None.
        spacing : array-like, optional
            3D array or list with grid spacing.
            Only required for ImageData grids.
            The default is None.

        Returns
        -------
        VtkFile
            This VtkFile to allow chained calls.
        """
        gType = self.ftype.name
        self.xml.openElement(gType)
        if gType == VtkImageData.name:
            if not start or not end or not origin or not spacing:
                assert False
            ext = _mix_extents(start, end)
            self.xml.addAttributes(
                WholeExtent=ext,
                Origin=_array_to_string(origin),
                Spacing=_array_to_string(spacing),
            )

        elif gType in [VtkStructuredGrid.name, VtkRectilinearGrid.name]:
            if not start or not end:
                assert False
            ext = _mix_extents(start, end)
            self.xml.addAttributes(WholeExtent=ext)

        return self

    def closeGrid(self):
        """
        Close grid element.

        Returns
        -------
        VtkFile
            This VtkFile to allow chained calls.
        """
        self.xml.closeElement(self.ftype.name)

    def addHeader(self, name, dtype, nelem, ncomp):
        """
        Add data array description to xml header section.

        Parameters
        ----------
        name : str
            data array name.
        dtype : str
            data type.
        nelem : int
            number of elements in array.
        ncomp : int
            number of components, 1 (=scalar) and 3 (=vector).

        Returns
        -------
        VtkFile
            This VtkFile to allow chained calls.

        Notes
        -----
        This is a low level function.
        Use addData if you want to add a numpy array.
        """
        dtype = np_to_vtk[dtype]

        self.xml.openElement("DataArray")
        self.xml.addAttributes(
            Name=name,
            NumberOfComponents=ncomp,
            type=dtype.name,
            format="appended",
            offset=self.offset,
        )
        self.xml.closeElement()

        self.offset += nelem * ncomp * dtype.size + 8  # add 8 to indicate array size
        return self

    def addData(self, name, data):
        """
        Add array description to xml header section.

        Parameters
        ----------
        name : str
            data array name.
        data : array-like
            one numpy array or a tuple with 3 numpy arrays.
            If a tuple, the individual arrays must represent the components
            of a vector field.
            All arrays must be one dimensional or three-dimensional.
        """
        if isinstance(data, tuple):  # vector data
            assert len(data) == 3
            x = data[0]
            self.addHeader(name, x.dtype.name, x.size, 3)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1 or data.ndim == 3:
                self.addHeader(name, data.dtype.name, data.size, 1)
            else:
                assert False, "Bad array shape: " + str(data.shape)
        else:
            assert False, "Argument must be a Numpy array"

    def appendHeader(self, dtype, nelem, ncomp):
        """
        Append size of data block to header.

        This function only writes the size of the data block
        that will be appended.
        The data itself must be written immediately after
        calling this function.

        Parameters
        ----------
        dtype : str
            string with data type representation (same as numpy).
            For example, 'float64'.
        nelem : int
            number of elements.
        ncomp : int
            number of components, 1 (=scalar) or 3 (=vector)..
        """
        self.openAppendedData()
        dsize = np_to_vtk[dtype].size
        block_size = dsize * ncomp * nelem
        writeBlockSize(self.xml.stream, block_size)
        # else:  # this routine does not exist!
        #     writeBlockSize64Bit(self.xml.stream, block_size)

    def appendData(self, data):
        """
        Append data to binary section.

        This function writes the header section
        and the data to the binary file.

        Parameters
        ----------
        data : array-like
            one numpy array or a tuple with 3 numpy arrays.
            If a tuple, the individual
            arrays must represent the components of a vector field.
            All arrays must be one dimensional or three-dimensional.
            The order of the arrays must coincide with
            the numbering scheme of the grid.

        Returns
        -------
        VtkFile
            This VtkFile to allow chained calls.
        """
        # TODO: Extend this function to accept contiguous C order arrays.
        self.openAppendedData()

        if isinstance(data, tuple):  # 3 numpy arrays
            ncomp = len(data)
            assert ncomp == 3
            dsize = data[0].dtype.itemsize
            nelem = data[0].size
            block_size = ncomp * nelem * dsize
            # if self.largeFile == False:
            writeBlockSize(self.xml.stream, block_size)
            # else:
            #    writeBlockSize64Bit(self.xml.stream, block_size)
            x, y, z = data[0], data[1], data[2]
            writeArraysToFile(self.xml.stream, x, y, z)

        elif isinstance(data, np.ndarray) and (
            data.ndim == 1 or data.ndim == 3
        ):  # single numpy array
            ncomp = 1
            dsize = data.dtype.itemsize
            nelem = data.size
            block_size = ncomp * nelem * dsize
            # if self.largeFile == False:
            writeBlockSize(self.xml.stream, block_size)
            # else:
            #    writeBlockSize64Bit(self.xml.stream, block_size)
            writeArrayToFile(self.xml.stream, data)

        else:
            assert False

        return self

    def openAppendedData(self):
        """
        Open binary section.

        It is not necessary to explicitly call this function
        from an external library.
        """
        if not self.appendedDataIsOpen:
            self.xml.openElement("AppendedData").addAttributes(encoding="raw").addText(
                "_"
            )
            self.appendedDataIsOpen = True

    def closeAppendedData(self):
        """
        Close binary section.

        It is not necessary to explicitly call this function
        from an external library.
        """
        self.xml.closeElement("AppendedData")

    def openElement(self, tagName):
        """
        Open an element.

        Useful to add elements such as: Coordinates, Points, Verts, etc.
        """
        self.xml.openElement(tagName)

    def closeElement(self, tagName):
        """Close an element."""
        self.xml.closeElement(tagName)

    def save(self):
        """Close file."""
        if self.appendedDataIsOpen:
            self.xml.closeElement("AppendedData")
        self.xml.closeElement("VTKFile")
        self.xml.close()


# ================================
#        VtkParallelFile class
# ================================
class VtkParallelFile:
    """
    Class for a VTK parallel file.

    Parameters
    ----------
    filepath : str
        filename without extension
    ftype : VtkParallelFileType
    """

    def __init__(self, filepath, ftype):
        assert isinstance(ftype, VtkParallelFileType)
        self.ftype = ftype
        self.filename = filepath + ftype.ext
        self.xml = XmlWriter(self.filename)
        self.xml.openElement("VTKFile").addAttributes(
            type=ftype.name,
            version="1.0",
            byte_order=_get_byte_order(),
            header_type="UInt64",
        )

    def getFileName(self):
        """Return absolute path to this file."""
        return os.path.abspath(self.filename)

    def addPiece(
        self,
        start=None,
        end=None,
        source=None,
    ):
        """
        Add piece section with extent and source.
        Parameters
        ----------
        start : array-like, optional
            array or list with start indexes in each direction.
            Must be given with end.
        end : array-like, optional
            array or list with end indexes in each direction.
            Must be given with start.
        source : str
            Source of this piece
        Returns
        -------
        VtkParallelFile
            This VtkFile to allow chained calls.
        """
        # Check Source
        assert source is not None
        assert source.split(".")[-1] == self.ftype.ext[2:]

        self.xml.openElement("Piece")
        if start and end:
            ext = _mix_extents(start, end)
            self.xml.addAttributes(Extent=ext)
        self.xml.addAttributes(Source=source)
        self.xml.closeElement()
        return self

    def openData(
        self,
        nodeType,
        scalars=None,
        vectors=None,
        normals=None,
        tensors=None,
        tcoords=None,
    ):
        """
        Open data section.
        Parameters
        ----------
        nodeType : str
            Either "Point", "Cell" or "Field".
        scalars : str, optional
            default data array name for scalar data.
        vectors : str, optional
            default data array name for vector data.
        normals : str, optional
            default data array name for normals data.
        tensors : str, optional
            default data array name for tensors data.
        tcoords : str, optional
            default data array name for tcoords data.
        Returns
        -------
        VtkFile
            This VtkFile to allow chained calls.
        """
        self.xml.openElement(nodeType + "Data")
        if scalars:
            self.xml.addAttributes(Scalars=scalars)
        if vectors:
            self.xml.addAttributes(Vectors=vectors)
        if normals:
            self.xml.addAttributes(Normals=normals)
        if tensors:
            self.xml.addAttributes(Tensors=tensors)
        if tcoords:
            self.xml.addAttributes(TCoords=tcoords)

        return self

    def closeData(self, nodeType):
        """
        Close data section.
        Parameters
        ----------
        nodeType : str
            "Point", "Cell" or "Field".
        Returns
        -------
        VtkFile
            This VtkFile to allow chained calls.
        """
        self.xml.closeElement(nodeType + "Data")

    def openGrid(self, start=None, end=None, origin=None, spacing=None, ghostlevel=0):
        """
        Open grid section.

        Parameters
        ----------
        start : array-like, optional
            array or list of start indexes.
            Required for Structured, Rectilinear and ImageData grids.
            The default is None.
        end : array-like, optional
            array or list of end indexes.
            Required for Structured, Rectilinear and ImageData grids.
            The default is None.
        origin : array-like, optional
            3D array or list with grid origin.
            Only required for ImageData grids.
            The default is None.
        spacing : array-like, optional
            3D array or list with grid spacing.
            Only required for ImageData grids.
            The default is None.
        ghostlevel : int
            Number of ghost-levels by which
            the extents in the individual pieces overlap.
        Returns
        -------
        VtkFile
            This VtkFile to allow chained calls.
        """
        gType = self.ftype.name
        self.xml.openElement(gType)

        if gType == VtkPImageData.name:
            if not start or not end or not origin or not spacing:
                raise ValueError(f"start, end, origin and spacing required for {gType}")
            ext = _mix_extents(start, end)
            self.xml.addAttributes(
                WholeExtent=ext,
                Origin=_array_to_string(origin),
                Spacing=_array_to_string(spacing),
            )

        elif gType in [VtkPStructuredGrid.name, VtkPRectilinearGrid.name]:
            if not start or not end:
                raise ValueError(f"start and end required for {gType}.")
            ext = _mix_extents(start, end)
            self.xml.addAttributes(WholeExtent=ext)

        # Ghostlevel
        self.xml.addAttributes(Ghostlevel=ghostlevel)
        return self

    def closeGrid(self):
        """
        Close grid element.
        Returns
        -------
        VtkFile
            This VtkFile to allow chained calls.
        """
        self.xml.closeElement(self.ftype.name)

    def addHeader(self, name, dtype, ncomp):
        """
        Add data array description to xml header section.
        Parameters
        ----------
        name : str
            data array name.
        dtype : str
            data type.
        ncomp : int
            number of components, 1 (=scalar) and 3 (=vector).
        Returns
        -------

        VtkFile
            This VtkFile to allow chained calls.
        Notes
        -----

        This is a low level function.
        Use addData if you want to add a numpy array.
        """
        dtype = np_to_vtk[dtype.name]

        self.xml.openElement("DataArray")
        self.xml.addAttributes(
            Name=name,
            NumberOfComponents=ncomp,
            type=dtype.name,
        )
        self.xml.closeElement()

    def openElement(self, tagName):
        """
        Open an element.
        Useful to add elements such as: Coordinates, Points, Verts, etc.
        """
        self.xml.openElement(tagName)

    def closeElement(self, tagName):
        self.xml.closeElement(tagName)

    def save(self):
        """Close file."""
        self.xml.closeElement("VTKFile")
        self.xml.close()
