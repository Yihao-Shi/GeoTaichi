import os 
import numpy as np
def fromat_worker(mpmfile:str,divide=True):
    """
    args:
        mpmfile[str]:mpmfile path
        divide[bool][option]:divide mpmfile into several file parts according material ID,then fromat to GEOTaichi input file.
            example(True):
                >>>divide_worker("mpmfile.part")
                output: 
                    mpmfile.part1
                    mpmfile.part2
                    mpmfile.part3
                    ...
                    mpmfile.part[material_ID]
            example(False):
                >>>divide_worker("mpmfile.part")
                output: 
                    mpmfile.partc
    """
    assert os.path.exists(mpmfile),f"file {mpmfile} not found"
    file_data = np.loadtxt(mpmfile,delimiter=' ',skiprows=3)
    file_data = move_data(file_data)
    min_x = np.min(file_data[:,1])
    min_y = np.min(file_data[:,2])
    min_z = np.min(file_data[:,3])
    max_x= np.max(file_data[:,1])
    max_y= np.max(file_data[:,2])
    max_z= np.max(file_data[:,3])
    print("min_x:",min_x,"min_y:",min_y,"min_z:",min_z)
    print("max_x:",max_x,"max_y:",max_y,"max_z:",max_z)
    print('Particle number:',file_data.shape[0])
    
    filedir = os.path.dirname(mpmfile)
    filename = os.path.basename(mpmfile).split('.')[0]
    #GEOTaichi input file format x,y,z,vol,psx,psy,psz
    #MPMgenerator input file format id x,y,z,vol,plp,material_ID
    matids = np.unique(file_data[:,6])
    print("Material number:",len(matids),"Material ID:",matids)
    if divide:
        for matid in matids:
            mask = file_data[:,6] == matid
            data = file_data[mask][:,[1,2,3,4,5,5,5,6]]
            np.savetxt(os.path.join(filedir,f"{filename}.part{int(matid)}"),data,delimiter=' ',fmt='%.6f',header = 'x y z vol psx psy psz matid',comments='#')
            print(f"save {filename}.part{int(matid)}")
    else:
        data = file_data[:,[1,2,3,4,5,5,5,6]]
        np.savetxt(os.path.join(filedir,f"{filename}.partc"),data,delimiter=' ',fmt='%.6f',header = 'x y z vol psx psy psz matid',comments='#')
def move_data(file_data:np.array):
    """
    平移坐标为正值
    """
    min_x = np.min(file_data[:,1])
    min_y = np.min(file_data[:,2])
    min_z = np.min(file_data[:,3])
    if min_x<0:
        file_data[:,1] -= min_x
    if min_y<0:
        file_data[:,2] -= min_y
    if min_z<0:
        file_data[:,3] -= min_z
    return file_data
    
if __name__ == '__main__':
    file =  'E:\\EX_library\\PAPER-3D_MODEL_MPM-231121\\MPM-Particle-Generator-1.1\\examples\\e3\\paper_model\\mpm_d2000.part'
    
    fromat_worker(file,1)