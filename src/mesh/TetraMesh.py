import numpy as np
import gmsh, tempfile

from src.mesh.GaussPoint import GaussPointInTriangle


class TetraMesh(object):
    def __init__(self) -> None:
        pass

    def local_gauss_point(self):
        gauss_point = GaussPointInTriangle()
        return gauss_point.gpcoords
    
    def generate_gauss_point(self, mesh, total_cell, file_name=None):
        gauss_coords = np.array([])
        points = np.array([])
        cells = np.array([])
        # checks mesher selection
        if mesher_id not in [1, 3, 4, 7, 9, 10]:
            raise ValueError("unavailable mesher selected!")
        else:
            mesher_id = int(mesher_id)

        # set max element length to a best guess if not specified
        if max_element is None:
            max_element = np.sqrt(np.mean(mesh.area_faces))

        if file_name is not None:
            # check extensions to make sure it is supported format
            if not any(
                file_name.lower().endswith(e)
                for e in [".bdf", ".msh", ".inp", ".diff", ".mesh"]
            ):
                raise ValueError(
                    "Only Nastran (.bdf), Gmsh (.msh), Abaqus (*.inp), "
                    + "Diffpack (*.diff) and Inria Medit (*.mesh) formats "
                    + "are available!"
                )

        # exports to disk for gmsh to read using a temp file
        mesh_file = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
        mesh_file.close()
        mesh.export(mesh_file.name)

        # starts Gmsh Python API script
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("Nastran_stl")

        gmsh.merge(mesh_file.name)
        dimtag = gmsh.model.getEntities()[0]
        dim = dimtag[0]
        tag = dimtag[1]

        surf_loop = gmsh.model.geo.addSurfaceLoop([tag])
        gmsh.model.geo.addVolume([surf_loop])
        gmsh.model.geo.synchronize()

        # We can then generate a 3D mesh...
        gmsh.option.setNumber("Mesh.Algorithm3D", mesher_id)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_element)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize(niter=2)
        numElementTypes = gmsh.model.mesh.getElementTypes().shape[0]
        while numElementTypes < total_cell:
            gmsh.model.mesh.refine()
            gmsh.model.mesh.optimize(niter=2)
            numElementTypes = gmsh.model.mesh.getElementTypes().shape[0]

        dimtag2 = gmsh.model.getEntities()[1]
        dim2 = dimtag2[0]
        tag2 = dimtag2[1]
        p2 = gmsh.model.addPhysicalGroup(dim2, [tag2])
        gmsh.model.setPhysicalName(dim, p2, "Nastran_bdf")

        # if file name is None, return msh data using a tempfile
        if not file_name is None:
            out_data = tempfile.NamedTemporaryFile(suffix=".msh", delete=False)
            # windows gets mad if two processes try to open the same file
            out_data.close()
            gmsh.write(out_data.name)

        elementTypes = gmsh.model.mesh.getElementTypes().shape[0]
        for t in elementTypes:
            localCoords, weights = gmsh.model.mesh.getIntegrationPoints(t, "Gauss1")
            jacobians, determinants, coords = gmsh.model.mesh.getJacobians(t, localCoords)
            gauss_coords = np.append(gauss_coords, coords)
        
        _, points, _ = gmsh.model.mesh.getNodes()
        _, cells, _ = gmsh.model.mesh.nei()
        # close up shop
        gmsh.finalize()
        return gauss_coords
    
    def get_jacobian(self):
        pass

    def global_gauss_point(self):
        pass

    def contruct(self):
        pass

    def is_boundary_point(self):
        pass

    def is_boundary_edge(self):
        pass

    def is_boundary_face(self):
        pass

    def is_boundary_cell(self):
        pass

    def boundary_point(self):
        pass

    def boundary_edge(self):
        pass

    def boundary_face(self):
        pass

    def boundary_cell(self):
        pass


