# check and draw the region of the image
# need install plotly 
# use pip install plotly or conda install plotly or (ubuntu)sudo apt-get install python3-plotly
import plotly.graph_objects as go
import numpy as np
import random
import os 
def change_StartPoint_EndPoint2BoundingBoxPoint_BoundingBoxSize(StartPoint,EndPoint):
    BoundingBoxPoint = np.array([StartPoint[0],StartPoint[1],StartPoint[2]])
    BoundingBoxSize = np.array([EndPoint[0]-StartPoint[0],EndPoint[1]-StartPoint[1],EndPoint[2]-StartPoint[2]])
    return BoundingBoxPoint,BoundingBoxSize
    
def draw_region(region):
    fig = go.Figure()
    if isinstance(region,dict):
        if region['Type'] == 'Rectangle':
            add_cube(fig,region['BoundingBoxPoint'].to_numpy(),region['BoundingBoxSize'].to_numpy(),region['Name'])
    elif isinstance(region,list):
        for i in region:
            if i['Type'] == 'Rectangle':
                add_cube(fig,i['BoundingBoxPoint'].to_numpy(),i['BoundingBoxSize'].to_numpy(),i['Name'])
    fig.update_layout(scene = dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'),
                  width=1000,
                  margin=dict(r=20, b=10, l=10, t=10))
    fig.show()
def draw_boundary(boundarys,show = False):
    count1 = 0
    fig = go.Figure()
    if isinstance(boundarys,dict):
        
        BoundingBoxPoint,BoundingBoxSize = change_StartPoint_EndPoint2BoundingBoxPoint_BoundingBoxSize(boundarys['StartPoint'],boundarys['EndPoint'])
        count1+=1
        add_cube(fig,np.array(BoundingBoxPoint),np.array(BoundingBoxSize),boundarys['BoundaryType']+f'{count1}')
    elif isinstance(boundarys,list):
        for i in boundarys:
            count1+=1
            BoundingBoxPoint,BoundingBoxSize = change_StartPoint_EndPoint2BoundingBoxPoint_BoundingBoxSize(i['StartPoint'],i['EndPoint'])
            add_cube(fig,np.array(BoundingBoxPoint),np.array(BoundingBoxSize),i['BoundaryType']+f'{count1}')
    fig.update_layout(scene = dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'),
                  width=1000,
                  margin=dict(r=20, b=10, l=10, t=10))
    if show:
        fig.show()
    else: return fig
def add_scatter(fig,pointsx,pointsy,pointsz,**kwargs): # 绘制物质点
    fig.add_trace(go.Scatter3d(x=pointsx, y=pointsy, z=pointsz,mode='markers',**kwargs))
    fig.show()
    
def draw_particle_cloud_with_boundary(particle_cloud,boundarys,**kwargs):
    fig = draw_boundary(boundarys,show = False)
    posx,posy,posz = particle_cloud[:,0],particle_cloud[:,1],particle_cloud[:,2]
    add_scatter(fig,posx,posy,posz,**kwargs)
# 定义长方体的8个顶点
def cuboid_data(base, size):
    o = np.array(base)
    l, w, h = size
    x = [o[0], o[0] + l]
    y = [o[1], o[1] + w]
    z = [o[2], o[2] + h]
    return np.array([[x[0], y[0], z[0]],
                     [x[1], y[0], z[0]],
                     [x[1], y[1], z[0]],
                     [x[0], y[1], z[0]],
                     [x[0], y[0], z[1]],
                     [x[1], y[0], z[1]],
                     [x[1], y[1], z[1]],
                     [x[0], y[1], z[1]]])
# 生成长方体的顶点和面
def create_cube(center, size):
    vertices = cuboid_data(center, size)
    faces = [[vertices[j] for j in [0,1,2,3]],
             [vertices[j] for j in [4,5,6,7]], 
             [vertices[j] for j in [0,3,7,4]], 
             [vertices[j] for j in [1,2,6,5]], 
             [vertices[j] for j in [0,1,5,4]], 
             [vertices[j] for j in [2,3,7,6]]]
    return faces
def add_cube(fig:go.Figure,base:list,size:list,name:str):
    base = base
    size = size

    faces = create_cube(base, size)
    # 提取所有顶点的坐标
    x, y, z = [], [], []
    for face in faces:
        for vertex in face:
            x.append(vertex[0])
            y.append(vertex[1])
            z.append(vertex[2])
    # 生成随机颜色
    r = random.random()
    g = random.random()
    b = random.random()
    color = f'rgb({r*255},{g*255},{b*255})'
    print("color:",color)
    fig.add_mesh3d(
        x=x,
        y=y,
        z=z,
        color=color,
        opacity=0.5,
        alphahull=0,
        name=name
    )
def MPM_particle_reader(MPM_class):
    scene = MPM_class.scene
    particle_num = scene.particleNum[0]
    particle_cloud = scene.particle.x.to_numpy()[0:scene.particleNum[0]]
    #posx = np.ascontiguousarray(position[:, 0])
    #posy = np.ascontiguousarray(position[:, 1])
    #posz = np.ascontiguousarray(position[:, 2])
    return particle_cloud
def load_particle_file(file_path):
    particle_cloud = np.loadtxt(file_path,skiprows =1) 
    #posx = data[:,0]
    #posy = data[:,1]
    #posz = data[:,2]
    return particle_cloud
def particle_cloud_filter( particle_cloud, region:dict):
    region = RE(region)
    if region.region_type == 'Rectangle':
        start_points = region.local_start_point
        size_points = region.local_region_size
        bounding_box = [start_points[0], start_points[0] + size_points[0], start_points[1], start_points[1] + size_points[1]]
        particle_cloud = particle_cloud[np.logical_and(particle_cloud[:, 0] >= bounding_box[0], particle_cloud[:, 0] <= bounding_box[1])]
        particle_cloud = particle_cloud[np.logical_and(particle_cloud[:, 1] >= bounding_box[2], particle_cloud[:, 1] <= bounding_box[3])]
        #if self.sims.dimension == 3:
        bounding_box += [start_points[2], start_points[2] + size_points[2]]
        particle_cloud = particle_cloud[np.logical_and(particle_cloud[:, 2] >= bounding_box[4], particle_cloud[:, 2] <= bounding_box[5])]
    return particle_cloud
def auto_boundary_limits(particle_cloud,region:dict,tol = 0.01): # tol = pisze/4
    region = RE(region)
    min_x ,max_x,min_y,max_y,min_z,max_z = min(particle_cloud[:, 0]),max(particle_cloud[:, 0]),min(particle_cloud[:, 1]),max(particle_cloud[:, 1]),min(particle_cloud[:, 2]),max(particle_cloud[:, 2])
    bounding_box = region.bounding_box
    if bounding_box[3]-max_y < tol:
        behind_SPointy = bounding_box[3]-tol
    else: behind_SPointy = max_y
    
    if min_y-bounding_box[2] < tol:
        front_EPointy = min_y+tol
    else: front_EPointy = min_y
    
    if min_x-bounding_box[0] < tol:
        left_EPointx = min_x+tol
    else: left_EPointx = min_x
    if bounding_box[1]-max_x < tol:
        right_SPointx = bounding_box[1]-tol
    else: right_SPointx = max_x 
    
    if min_z-bounding_box[4] < tol:
        bottom_EPointz = min_z+tol
    else: bottom_EPointz = min_z
    if bounding_box[5]-max_z < tol:
        top_SPointz = bounding_box[5]-tol
    else: top_SPointz = max_z
    return_dict = {
        'behind': {
            'StartPoint': [bounding_box[0],behind_SPointy ,bounding_box[4]],
            'EndPoint':   [bounding_box[1],bounding_box[3],bounding_box[5]]
            },
        'front': {
            'StartPoint': [bounding_box[0],bounding_box[2],bounding_box[4]],
            'EndPoint':   [bounding_box[1],front_EPointy  ,bounding_box[5]]
            },
        'left': {
            'StartPoint': [bounding_box[0],bounding_box[2],bounding_box[4]],
            'EndPoint':   [left_EPointx,bounding_box[3],bounding_box[5]]
            },
        'right': {
            'StartPoint': [right_SPointx,bounding_box[2],bounding_box[4]],
            'EndPoint':   [bounding_box[1],bounding_box[3],bounding_box[5]]
        },
        'bottom': {
            'StartPoint': [bounding_box[0],bounding_box[2],bounding_box[4]],
            'EndPoint':   [bounding_box[1],bounding_box[3],bottom_EPointz]
            },
        'top': {
            'StartPoint': [bounding_box[0],bounding_box[2],top_SPointz],
            'EndPoint':   [bounding_box[1],bounding_box[3],bounding_box[5]]
        }
    }
    return return_dict

class RE:
    def __init__(self,region_dict = {}):
        self.name               = region_dict["Name"]
        self.region_type        = region_dict["Type"]
        self.local_start_point  = region_dict["BoundingBoxPoint"]
        self.local_region_size  = region_dict["BoundingBoxSize"]
        self.bounding_box = [self.local_start_point[0], self.local_start_point[0] + self.local_region_size[0], self.local_start_point[1], self.local_start_point[1] + self.local_region_size[1]]
        if len(self.local_start_point) == 3:
            self.bounding_box += [self.local_start_point[2], self.local_start_point[2] + self.local_region_size[2]]
        