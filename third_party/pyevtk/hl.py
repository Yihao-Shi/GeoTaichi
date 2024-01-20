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
"""High level Python library to export data to binary VTK file."""

import numpy as np
from .vtk import (
    VtkFile,
    VtkParallelFile,
    VtkUnstructuredGrid,
    VtkImageData,
    VtkRectilinearGrid,
    VtkStructuredGrid,
    VtkPImageData,
    VtkPRectilinearGrid,
    VtkPStructuredGrid,
    VtkUnstructuredGrid,
    VtkVertex,
    VtkLine,
    VtkPolyLine,
    VtkPixel,
)


# =================================
#       Helper functions
# =================================
def _addDataToFile(vtkFile, cellData, pointData, fieldData=None):
    # Point data
    if pointData:
        keys = list(pointData.keys())
        # find first scalar and vector data key to set it as attribute
        scalars = next(
            (key for key in keys if isinstance(pointData[key], np.ndarray)), None
        )
        vectors = next((key for key in keys if isinstance(pointData[key], tuple)), None)
        vtkFile.openData("Point", scalars=scalars, vectors=vectors)
        for key in keys:
            data = pointData[key]
            vtkFile.addData(key, data)
        vtkFile.closeData("Point")

    # Cell data
    if cellData:
        keys = list(cellData.keys())
        # find first scalar and vector data key to set it as attribute
        scalars = next(
            (key for key in keys if isinstance(cellData[key], np.ndarray)), None
        )
        vectors = next((key for key in keys if isinstance(cellData[key], tuple)), None)
        vtkFile.openData("Cell", scalars=scalars, vectors=vectors)
        for key in keys:
            data = cellData[key]
            vtkFile.addData(key, data)
        vtkFile.closeData("Cell")

    # Field data
    # https://www.visitusers.org/index.php?title=Time_and_Cycle_in_VTK_files#XML_VTK_files
    if fieldData:
        keys = list(fieldData.keys())
        vtkFile.openData("Field")  # no attributes in FieldData
        for key in keys:
            data = fieldData[key]
            vtkFile.addData(key, data)
        vtkFile.closeData("Field")


def _addDataToParallelFile(vtkParallelFile, cellData, pointData):
    assert isinstance(vtkParallelFile, VtkParallelFile)
    # Point data
    if pointData:
        keys = list(pointData.keys())
        # find first scalar and vector data key to set it as attribute
        scalars = next((key for key in keys if pointData[key][1] == 1), None)
        vectors = next((key for key in keys if pointData[key][1] == 3), None)
        vtkParallelFile.openData("PPoint", scalars=scalars, vectors=vectors)
        for key in keys:
            dtype, ncomp = pointData[key]
            vtkParallelFile.addHeader(key, dtype=dtype, ncomp=ncomp)
        vtkParallelFile.closeData("PPoint")

    # Cell data
    if cellData:
        keys = list(cellData.keys())
        # find first scalar and vector data key to set it as attribute
        scalars = next((key for key in keys if cellData[key][1] == 1), None)
        vectors = next((key for key in keys if cellData[key][1] == 3), None)
        vtkParallelFile.openData("PCell", scalars=scalars, vectors=vectors)
        for key in keys:
            dtype, ncomp = cellData[key]
            vtkParallelFile.addHeader(key, dtype=dtype, ncomp=ncomp)
        vtkParallelFile.closeData("PCell")


def _appendDataToFile(vtkFile, cellData, pointData, fieldData=None):
    # Append data to binary section
    if pointData is not None:
        keys = list(pointData.keys())
        for key in keys:
            data = pointData[key]
            vtkFile.appendData(data)

    if cellData is not None:
        keys = list(cellData.keys())
        for key in keys:
            data = cellData[key]
            vtkFile.appendData(data)

    if fieldData is not None:
        keys = list(fieldData.keys())
        for key in keys:
            data = fieldData[key]
            vtkFile.appendData(data)


# =================================
#       High level functions
# =================================
def imageToVTK(
    path,
    origin=(0.0, 0.0, 0.0),
    spacing=(1.0, 1.0, 1.0),
    cellData=None,
    pointData=None,
    fieldData=None,
    start=(0, 0, 0),
):
    """
    Export data values as a rectangular image.

    Parameters
    ----------
    path : str
        name of the file without extension where data should be saved.
    start : tuple, optional
        start of the coordinates.
        Used in the distributed context where each process
        writes its own vtk file. Default is (0, 0, 0).
    origin : tuple, optional
        grid origin.
        The default is (0.0, 0.0, 0.0).
    spacing : tuple, optional
        grid spacing.
        The default is (1.0, 1.0, 1.0).
    cellData : dict, optional
        dictionary containing arrays with cell centered data.
        Keys should be the names of the data arrays.
        Arrays must have the same dimensions in all directions and can contain
        scalar data ([n,n,n]) or vector data ([n,n,n],[n,n,n],[n,n,n]).
        The default is None.
    pointData : dict, optional
        dictionary containing arrays with node centered data.
        Keys should be the names of the data arrays.
        Arrays must have same dimension in each direction and
        they should be equal to the dimensions of the cell data plus one and
        can contain scalar data ([n+1,n+1,n+1]) or
        ([n+1,n+1,n+1],[n+1,n+1,n+1],[n+1,n+1,n+1]).
        The default is None.
    fieldData : dict, optional
        dictionary with variables associated with the field.
        Keys should be the names of the variable stored in each array.

    Returns
    -------
    str
        Full path to saved file.

    Notes
    -----
    At least, cellData or pointData must be present
    to infer the dimensions of the image.
    """
    assert cellData is not None or pointData is not None

    # Extract dimensions
    end = None
    if cellData is not None:
        keys = list(cellData.keys())
        data = cellData[keys[0]]
        if hasattr(data, "shape"):
            end = data.shape
        elif data[0].ndim == 3 and data[1].ndim == 3 and data[2].ndim == 3:
            end = data[0].shape
    elif pointData is not None:
        keys = list(pointData.keys())
        data = pointData[keys[0]]
        if hasattr(data, "shape"):
            end = data.shape
        elif data[0].ndim == 3 and data[1].ndim == 3 and data[2].ndim == 3:
            end = data[0].shape
        end = (end[0] - 1, end[1] - 1, end[2] - 1)

    # Write data to file
    w = VtkFile(path, VtkImageData)
    w.openGrid(start=start, end=end, origin=origin, spacing=spacing)
    w.openPiece(start=start, end=end)
    _addDataToFile(w, cellData, pointData, fieldData)
    w.closePiece()
    w.closeGrid()
    _appendDataToFile(w, cellData, pointData, fieldData)
    w.save()
    return w.getFileName()


# ==============================================================================
def gridToVTK(
    path, x, y, z, cellData=None, pointData=None, fieldData=None, start=(0, 0, 0)
):
    """
    Write data values as a rectilinear or structured grid.

    Parameters
    ----------
    path : str
        name of the file without extension where data should be saved.
    x : array-like
        x coordinate axis.
    y : array-like
        y coordinate axis.
    z : array-like
        z coordinate axis.
    start : tuple, optional
        start of the coordinates.
        Used in the distributed context where each process
        writes its own vtk file. Default is (0, 0, 0).
    cellData : dict, optional
        dictionary containing arrays with cell centered data.
        Keys should be the names of the data arrays.
        Arrays must have the same dimensions in all directions and must contain
        only scalar data.
    pointData : dict, optional
        dictionary containing arrays with node centered data.
        Keys should be the names of the data arrays.
        Arrays must have same dimension in each direction and
        they should be equal to the dimensions of the cell data plus one and
        must contain only scalar data.
    fieldData : dict, optional
        dictionary with variables associated with the field.
        Keys should be the names of the variable stored in each array.
    Returns
    -------
    str
        Full path to saved file.
    Notes
    -----
    coordinates of the nodes of the grid. They can be 1D or 3D depending if
    the grid should be saved as a rectilinear or logically structured grid,
    respectively.
    Arrays should contain coordinates of the nodes of the grid.
    If arrays are 1D, then the grid should be Cartesian,
    i.e. faces in all cells are orthogonal.
    If arrays are 3D, then the grid should be logically structured
    with hexahedral cells.
    In both cases the arrays dimensions should be
    equal to the number of nodes of the grid.
    """
    nx = ny = nz = 0

    if x.ndim == 1 and y.ndim == 1 and z.ndim == 1:
        nx, ny, nz = x.size - 1, y.size - 1, z.size - 1
        isRect = True
        ftype = VtkRectilinearGrid
    elif x.ndim == 3 and y.ndim == 3 and z.ndim == 3:
        s = x.shape
        nx, ny, nz = s[0] - 1, s[1] - 1, s[2] - 1
        isRect = False
        ftype = VtkStructuredGrid
    else:
        raise ValueError(
            f"x, y and z should have ndim == 3 but they have ndim of {x.ndim}, {y.ndim}"
            f" and {z.ndim} respectively"
        )

    # Write extent
    end = (start[0] + nx, start[1] + ny, start[2] + nz)

    # Open File
    w = VtkFile(path, ftype)

    # Open Grid part
    w.openGrid(start=start, end=end)
    w.openPiece(start=start, end=end)

    # Add coordinates description
    if isRect:
        w.openElement("Coordinates")
        w.addData("x_coordinates", x)
        w.addData("y_coordinates", y)
        w.addData("z_coordinates", z)
        w.closeElement("Coordinates")
    else:
        w.openElement("Points")
        w.addData("points", (x, y, z))
        w.closeElement("Points")

    # Add data description
    _addDataToFile(w, cellData, pointData, fieldData)

    # Close Grid part
    w.closePiece()
    w.closeGrid()

    # Write coordinates
    if isRect:
        w.appendData(x).appendData(y).appendData(z)
    else:
        w.appendData((x, y, z))
    # Write data
    _appendDataToFile(w, cellData, pointData, fieldData)

    # Close file
    w.save()

    return w.getFileName()


def writeParallelVTKGrid(
    path, coordsData, starts, ends, sources, ghostlevel=0, cellData=None, pointData=None
):
    """
    Writes a parallel vtk file from grid-like data:
    VTKStructuredGrid or VTKRectilinearGrid

    Parameters
    ----------
    path : str
        name of the file without extension.
    coordsData : tuple
        2-tuple (shape, dtype) where shape is the
        shape of the coordinates of the full mesh
        and dtype is the dtype of the coordinates.
    starts : list
        list of 3-tuple representing where each source file starts
        in each dimension
    source : list
        list of the relative paths of the source files where the actual data is found
    ghostlevel : int, optional
        Number of ghost-levels by which
        the extents in the individual source files overlap.
    pointData : dict
        dictionnary containing the information about the arrays
        containing node centered data.
        Keys shoud be the names of the arrays.
        Values are (dtype, number of components)
    cellData :
        dictionnary containing the information about the arrays
        containing cell centered data.
        Keys shoud be the names of the arrays.
        Values are (dtype, number of components)
    """
    # Check that every source as a start and an end
    assert len(starts) == len(ends) == len(sources)

    # Get the extension + check that it's consistent accros all source files
    common_ext = sources[0].split(".")[-1]
    assert all(s.split(".")[-1] == common_ext for s in sources)

    if common_ext == "vts":
        ftype = VtkPStructuredGrid
        is_Rect = False
    elif common_ext == "vtr":
        ftype = VtkPRectilinearGrid
        is_Rect = True
    else:
        raise ValueError("This functions is meant to work only with ")

    w = VtkParallelFile(path, ftype)
    start = (0, 0, 0)
    (s_x, s_y, s_z), dtype = coordsData
    end = s_x - 1, s_y - 1, s_z - 1

    w.openGrid(start=start, end=end, ghostlevel=ghostlevel)

    _addDataToParallelFile(w, cellData=cellData, pointData=pointData)

    if is_Rect:
        w.openElement("PCoordinates")
        w.addHeader("x_coordinates", dtype=dtype, ncomp=1)
        w.addHeader("y_coordinates", dtype=dtype, ncomp=1)
        w.addHeader("z_coordinates", dtype=dtype, ncomp=1)
        w.closeElement("PCoordinates")
    else:
        w.openElement("PPoints")
        w.addHeader("points", dtype=dtype, ncomp=3)
        w.closeElement("PPoints")

    for start_source, end_source, source in zip(starts, ends, sources):
        w.addPiece(start_source, end_source, source)

    w.closeGrid()
    w.save()
    return w.getFileName()


# ==============================================================================
def pointsToVTK(path, x, y, z, data=None, fieldData=None):
    """
    Export points and associated data as an unstructured grid.

    Parameters
    ----------
    path : str
        name of the file without extension where data should be saved.
    x : array-like
        x coordinates of the points.
    y : array-like
        y coordinates of the points.
    z : array-like
        z coordinates of the points.
    data : dict, optional
        dictionary with variables associated to each point.
        Keys should be the names of the variable stored in each array.
        All arrays must have the same number of elements.
    fieldData : dict, optional
        dictionary with variables associated with the field.
        Keys should be the names of the variable stored in each array.

    Returns
    -------
    str
        Full path to saved file.
    """
    assert x.size == y.size == z.size
    npoints = x.size

    # create some temporary arrays to write grid topology
    offsets = np.arange(
        start=1, stop=npoints + 1, dtype="int32"
    )  # index of last node in each cell
    connectivity = np.arange(
        npoints, dtype="int32"
    )  # each point is only connected to itself
    cell_types = np.empty(npoints, dtype="uint8")

    cell_types[:] = VtkVertex.tid

    w = VtkFile(path, VtkUnstructuredGrid)
    w.openGrid()
    w.openPiece(ncells=npoints, npoints=npoints)

    w.openElement("Points")
    w.addData("points", (x, y, z))
    w.closeElement("Points")
    w.openElement("Cells")
    w.addData("connectivity", connectivity)
    w.addData("offsets", offsets)
    w.addData("types", cell_types)
    w.closeElement("Cells")

    _addDataToFile(w, cellData=None, pointData=data, fieldData=fieldData)

    w.closePiece()
    w.closeGrid()
    w.appendData((x, y, z))
    w.appendData(connectivity).appendData(offsets).appendData(cell_types)

    _appendDataToFile(w, cellData=None, pointData=data, fieldData=fieldData)

    w.save()
    return w.getFileName()


# ==============================================================================
def linesToVTK(path, x, y, z, cellData=None, pointData=None, fieldData=None):
    """
    Export line segments that joint 2 points and associated data.

    Parameters
    ----------
    path : str
        name of the file without extension where data should be saved.
    x : array-like
        x coordinates of the points in lines.
    y : array-like
        y coordinates of the points in lines.
    z : array-like
        z coordinates of the points in lines.
    cellData : dict, optional
        dictionary with variables associated to each line.
        Keys should be the names of the variable stored in each array.
        All arrays must have the same number of elements.
    pointData : dict, optional
        dictionary with variables associated to each vertex.
        Keys should be the names of the variable stored in each array.
        All arrays must have the same number of elements.
    fieldData : dict, optional
        dictionary with variables associated with the field.
        Keys should be the names of the variable stored in each array.

    Returns
    -------
    str
        Full path to saved file.

    Notes
    -----
    x, y, z are 1D arrays with coordinates of the vertex of the lines.
    It is assumed that each line.
    is defined by two points,
    then the lenght of the arrays should be equal to 2 * number of lines.
    """
    assert x.size == y.size == z.size
    assert x.size % 2 == 0
    npoints = x.size
    ncells = x.size / 2

    # Check cellData has the same size that the number of cells

    # create some temporary arrays to write grid topology
    offsets = np.arange(
        start=2, step=2, stop=npoints + 1, dtype="int32"
    )  # index of last node in each cell
    connectivity = np.arange(
        npoints, dtype="int32"
    )  # each point is only connected to itself
    cell_types = np.empty(npoints, dtype="uint8")

    cell_types[:] = VtkLine.tid

    w = VtkFile(path, VtkUnstructuredGrid)
    w.openGrid()
    w.openPiece(ncells=ncells, npoints=npoints)

    w.openElement("Points")
    w.addData("points", (x, y, z))
    w.closeElement("Points")
    w.openElement("Cells")
    w.addData("connectivity", connectivity)
    w.addData("offsets", offsets)
    w.addData("types", cell_types)
    w.closeElement("Cells")

    _addDataToFile(w, cellData=cellData, pointData=pointData, fieldData=fieldData)

    w.closePiece()
    w.closeGrid()
    w.appendData((x, y, z))
    w.appendData(connectivity).appendData(offsets).appendData(cell_types)

    _appendDataToFile(w, cellData=cellData, pointData=pointData, fieldData=fieldData)

    w.save()
    return w.getFileName()


# ==============================================================================
def polyLinesToVTK(
    path, x, y, z, pointsPerLine, cellData=None, pointData=None, fieldData=None
):
    """
    Export line segments that joint n points and associated data.

    Parameters
    ----------
    path : str
        name of the file without extension where data should be saved.
    x : array-like
        x coordinates of the points in lines.
    y : array-like
        y coordinates of the points in lines.
    z : array-like
        z coordinates of the points in lines.
    pointsPerLine : array-like
        Points in each poly-line.
    cellData : dict, optional
        dictionary with variables associated to each line.
        Keys should be the names of the variable stored in each array.
        All arrays must have the same number of elements.
    pointData : dict, optional
        dictionary with variables associated to each vertex.
        Keys should be the names of the variable stored in each array.
        All arrays must have the same number of elements.
    fieldData : dict, optional
        dictionary with variables associated with the field.
        Keys should be the names of the variable stored in each array.

    Returns
    -------
    str
        Full path to saved file.
    """
    assert x.size == y.size == z.size
    npoints = x.size
    ncells = pointsPerLine.size

    # create some temporary arrays to write grid topology
    offsets = np.zeros(ncells, dtype="int32")  # index of last node in each cell
    ii = 0
    for i in range(ncells):
        ii += pointsPerLine[i]
        offsets[i] = ii

    connectivity = np.arange(
        npoints, dtype="int32"
    )  # each line connects points that are consecutive

    cell_types = np.empty(npoints, dtype="uint8")
    cell_types[:] = VtkPolyLine.tid

    w = VtkFile(path, VtkUnstructuredGrid)
    w.openGrid()
    w.openPiece(ncells=ncells, npoints=npoints)

    w.openElement("Points")
    w.addData("points", (x, y, z))
    w.closeElement("Points")
    w.openElement("Cells")
    w.addData("connectivity", connectivity)
    w.addData("offsets", offsets)
    w.addData("types", cell_types)
    w.closeElement("Cells")

    _addDataToFile(w, cellData=cellData, pointData=pointData, fieldData=fieldData)

    w.closePiece()
    w.closeGrid()
    w.appendData((x, y, z))
    w.appendData(connectivity).appendData(offsets).appendData(cell_types)

    _appendDataToFile(w, cellData=cellData, pointData=pointData, fieldData=fieldData)

    w.save()
    return w.getFileName()


# ==============================================================================
def unstructuredGridToVTK(
    path,
    x,
    y,
    z,
    connectivity,
    offsets,
    cell_types,
    cellData=None,
    pointData=None,
    fieldData=None,
):
    """
    Export unstructured grid and associated data.

    Parameters
    ----------
    path : str
        name of the file without extension where data should be saved.
    x : array-like
        x coordinates of the vertices.
    y : array-like
        y coordinates of the vertices.
    z : array-like
        z coordinates of the vertices.
    connectivity : array-like
        1D array that defines the vertices associated to each element.
        Together with offset define the connectivity or topology of the grid.
        It is assumed that vertices in an element are listed consecutively.
    offsets : array-like
        1D array with the index of the last vertex of each element
        in the connectivity array.
        It should have length nelem,
        where nelem is the number of cells or elements in the grid..
    cell_types : TYPE
        1D array with an integer that defines the cell type of
        each element in the grid.
        It should have size nelem.
        This should be assigned from evtk.vtk.VtkXXXX.tid, where XXXX represent
        the type of cell.
        Please check the VTK file format specification for allowed cell types.
    cellData : dict, optional
        dictionary with variables associated to each cell.
        Keys should be the names of the variable stored in each array.
        All arrays must have the same number of elements.
    pointData : dict, optional
        dictionary with variables associated to each vertex.
        Keys should be the names of the variable stored in each array.
        All arrays must have the same number of elements.
    fieldData : dict, optional
        dictionary with variables associated with the field.
        Keys should be the names of the variable stored in each array.

    Returns
    -------
    str
        Full path to saved file.
    """
    assert x.size == y.size == z.size
    npoints = x.size
    ncells = cell_types.size
    assert offsets.size == ncells

    w = VtkFile(path, VtkUnstructuredGrid)
    w.openGrid()
    w.openPiece(ncells=ncells, npoints=npoints)

    w.openElement("Points")
    w.addData("points", (x, y, z))
    w.closeElement("Points")
    w.openElement("Cells")
    w.addData("connectivity", connectivity)
    w.addData("offsets", offsets)
    w.addData("types", cell_types)
    w.closeElement("Cells")

    _addDataToFile(w, cellData=cellData, pointData=pointData, fieldData=fieldData)

    w.closePiece()
    w.closeGrid()
    w.appendData((x, y, z))
    w.appendData(connectivity).appendData(offsets).appendData(cell_types)

    _appendDataToFile(w, cellData=cellData, pointData=pointData, fieldData=fieldData)

    w.save()
    return w.getFileName()


# ==============================================================================
def cylinderToVTK(
    path,
    x0,
    y0,
    z0,
    z1,
    radius,
    nlayers,
    npilars=16,
    cellData=None,
    pointData=None,
    fieldData=None,
):
    """
    Export cylinder as VTK unstructured grid.

    Parameters
    ----------
    path : str
        name of the file without extension where data should be saved.
    x0 : float
        x-center of the cylinder.
    y0 : float
        y-center of the cylinder.
    z0 : float
        lower end of the cylinder.
    z1 : float
        upper end of the cylinder.
    radius : float
        radius of the cylinder.
    nlayers : int
        Number of layers in z direction to divide the cylinder..
    npilars : int, optional
        Number of points around the diameter of the cylinder.
        Higher value gives higher resolution to represent the curved shape.
        The default is 16.
    cellData : dict, optional
        dictionary with variables associated to each cell.
        Keys should be the names of the variable stored in each array.
        Arrays should have number of elements equal to
        ncells = npilars * nlayers.
    pointData : dict, optional
        dictionary with variables associated to each vertex.
        Keys should be the names of the variable stored in each array.
        Arrays should have number of elements equal to
        npoints = npilars * (nlayers + 1).
    fieldData : dict, optional
        dictionary with variables associated with the field.
        Keys should be the names of the variable stored in each array.

    Returns
    -------
    str
        Full path to saved file.

    Notes
    -----
    This function only export vertical shapes for now.
    However, it should be easy to
    rotate the cylinder to represent other orientations.
    """
    import math as m

    # Define x, y coordinates from polar coordinates.
    dpi = 2.0 * m.pi / npilars
    angles = np.arange(0.0, 2.0 * m.pi, dpi)

    x = radius * np.cos(angles) + x0
    y = radius * np.sin(angles) + y0

    dz = (z1 - z0) / nlayers
    z = np.arange(z0, z1 + dz, step=dz)

    npoints = npilars * (nlayers + 1)
    ncells = npilars * nlayers

    xx = np.zeros(npoints)
    yy = np.zeros(npoints)
    zz = np.zeros(npoints)

    ii = 0
    for k in range(nlayers + 1):
        for p in range(npilars):
            xx[ii] = x[p]
            yy[ii] = y[p]
            zz[ii] = z[k]
            ii = ii + 1

    # Define connectivity
    conn = np.zeros(4 * ncells, dtype=np.int64)
    ii = 0
    for l in range(nlayers):
        for p in range(npilars):
            p0 = p
            if p + 1 == npilars:
                p1 = 0
            else:
                p1 = p + 1  # circular loop

            n0 = p0 + l * npilars
            n1 = p1 + l * npilars
            n2 = n0 + npilars
            n3 = n1 + npilars

            conn[ii + 0] = n0
            conn[ii + 1] = n1
            conn[ii + 2] = n3
            conn[ii + 3] = n2
            ii = ii + 4

    # Define offsets
    offsets = np.zeros(ncells, dtype=np.int64)
    for i in range(ncells):
        offsets[i] = (i + 1) * 4

    # Define cell types
    ctype = np.ones(ncells) + VtkPixel.tid

    return unstructuredGridToVTK(
        path,
        xx,
        yy,
        zz,
        connectivity=conn,
        offsets=offsets,
        cell_types=ctype,
        cellData=cellData,
        pointData=pointData,
        fieldData=fieldData,
    )
