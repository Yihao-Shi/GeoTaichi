import os
import numpy as np
def F90_file_loader(file_path:str,save_file:str):
    """将F90输入文件转为MPMgenerator生成的mpmfile文件"""
    #read file using os
#read file using os
    encoding = 'utf-8'
    try:
        with open(file_path,'r',encoding=encoding) as f:
            line = f.readline()
    except:
        encoding = 'gbk'
        with open(file_path,'r',encoding=encoding) as f:
            line = f.readline()
    write_flac = False
    body1 = 0
    Total_points = 0
    temp_file = save_file
    if os.path.exists(temp_file):
        os.remove(temp_file)
    #write title to temp file
    with open(temp_file,'a') as temp:
        temp.write('id x y z vol lp matid bodyid\n')
        temp.write('%PARTICLES\n')
        temp.write('unknown\n')
    with open(file_path,'r',encoding=encoding) as f:
        while True:
            line = f.readline()
            if not line:
                print("end of file")
                break
            line = line.strip()
            if 'dcell' in line or 'DCELL' in line: #dcell  2.0 read 2.0 as dcell
                dcell = float(line.split()[1])
            if 'Particle point' in line or 'PARTICLE POINT' in line: #Particle point 1 3770 use 1 as frc_id ,   use 3770 as npoints
                write_flac = True
                body1 += 1
                frc_id = int(line.split()[2])
                npoints = int(line.split()[3])
                Total_points += npoints
                print("write_flac:",write_flac,"frc_id:",frc_id,"npoints:",npoints)
                continue
            if write_flac and npoints:
                if 'load' in line or 'LOAD' in line:
                    write_flac =False
                    break
                with open(temp_file,'a') as temp:
                    id = int(line.split()[0])
                    matid = int(line.split()[1])
                    density = float(line.split()[2])
                    x = float(line.split()[3])
                    y = float(line.split()[4])
                    z = float(line.split()[5])
                    pxyz = dcell/2
                    vol = pxyz**3
                    temp.write(f'{id} '+f'{x} '+ f'{y} '+f'{z} '+f'{vol} '+f'{pxyz} '+f'{matid} '+f'{body1}'    +'\n')
                npoints -= 1
    #更改第三行的内容为颗粒数量
    with open(temp_file,'r') as f:
        lines = f.readlines()
    lines[2] = f'{Total_points}\n'
    with open(temp_file,'w') as f:
        f.writelines(lines)


