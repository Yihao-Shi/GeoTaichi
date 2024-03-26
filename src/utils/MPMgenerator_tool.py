import os 
import numpy as np
def divide_worker(mpmfile:str):
    """
    divide mpmfile into several file parts according material ID,then fromat to GEOTaichi input file.
    example:
        >>>divide_worker("mpmfile.part")
        output: 
            mpmfile.part1
            mpmfile.part2
            mpmfile.part3
            ...
            mpmfile.part[material_ID]
    """
    assert os.path.exists(mpmfile),f"file {mpmfile} not found"
    file_data = np.loadtxt(mpmfile,delimiter=' ',skiprows=3)
    filedir = os.path.dirname(mpmfile)
    filename = os.path.basename(mpmfile).split('.')[0]
    #GEOTaichi input file format x,y,z,vol,psx,psy,psz
    #MPMgenerator input file format id x,y,z,vol,plp,material_ID
    matids = np.unique(file_data[:,6])
    for matid in matids:
        mask = file_data[:,6] == matid
        data = file_data[mask][:,[1,2,3,4,5,5,5,6]]
        np.savetxt(os.path.join(filedir,f"{filename}.part{int(matid)}"),data,delimiter=' ',fmt='%.6f',header = 'x y z vol psx psy psz matid',comments='#')

if __name__ == '__main__':
    file =  'E:\\EX_library\\PAPER-3D_MODEL_MPM-231121\\MPM-Particle-Generator-1.    1\\examples\\example-2-daguangbao\\paper_model\\python_mpm_50000.part'
    
    divide_worker(file)