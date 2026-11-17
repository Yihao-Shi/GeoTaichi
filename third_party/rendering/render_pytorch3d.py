import math
import torch
import trimesh
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from pytorch3d.io import IO
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import TexturesVertex, look_at_view_transform, FoVPerspectiveCameras
from pytorch3d.renderer import PointLights, RasterizationSettings, MeshRenderer, PointsRenderer, PointsRasterizer, PointsRasterizationSettings
from pytorch3d.renderer import MeshRasterizer, SoftPhongShader, PerspectiveCameras, AlphaCompositor
from pytorch3d.io import load_objs_as_meshes, load_obj
import os
import glob
import tqdm
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams
import numpy as np

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# a simple shader that only renders the raw colors of the vertices
class SimpleShader(torch.nn.Module):
    def __init__(self, device="cpu", blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images  # (N, H, W, 3) RGBA image

def load_glb_mesh(file_path, normalize=True):
    mesh = trimesh.load(file_path, force='mesh')
    
    return mesh


def load_mesh(file_path, rotate=None, normalize=True):
    mesh = trimesh.load(file_path)
    # return mesh
    vertex_colors = torch.tensor(mesh.visual.vertex_colors[:,:3], dtype=torch.float).unsqueeze(0).to(device)
    vertices = torch.tensor(mesh.vertices, dtype=torch.float).unsqueeze(0).to(device)

    if normalize:
        # normalize the mesh to fit in a unit sphere
        max_dist = torch.max(torch.norm(vertices, dim=-1, keepdim=True))
        expand_scale = 1.0 / max_dist
    else:
        expand_scale = 1.0

    vertices = vertices * expand_scale
    faces = torch.tensor(mesh.faces, dtype=torch.int).unsqueeze(0).to(device)

    if rotate=='ours':
        vertices = vertices @ torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float).to(device)
    elif rotate=='imesh':
        vertices = vertices @ torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, -1]], dtype=torch.float).to(device)
        vertices = vertices @ torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float).to(device)

    
    textures = TexturesVertex(verts_features=vertex_colors).to(device)
    return Meshes(vertices, faces, textures).to(device)


# writes a fundction to load obj mesh with mtl
def load_obj_mesh(file_path, rotate=None, normalize=True):
    verts, faces, aux = load_obj(file_path)
    # meshes = load_objs_as_meshes([file_path], device=device, load_textures=True)
    return meshes

def get_render_cameras(FOV, ele=20, radius=1.3, nb=None):
    cameras = []
    nb = 6 if nb is None else nb
    for i in range(nb):
        R, T = look_at_view_transform(radius, ele, 360 / nb * i)
        cameras.append(FoVPerspectiveCameras(device=device, R=R, T=T, fov=FOV))
    # for j in range(6):
    #     R, T = look_at_view_transform(1.3, 20, 360 / 6 * j + 30)
    #     cameras.append(FoVPerspectiveCameras(device=device, R=R, T=T, fov=FOV))

    return cameras

def get_render_cameras_ellipse(FOV, ele_range=[-30,30], az_range=[-30,30], radius=1.3, nb=120):
    cameras = []
    eles_a = np.linspace(ele_range[0], ele_range[1], nb // 2)
    eles_b = np.linspace(ele_range[1], ele_range[0], nb // 2)
    eles = np.concatenate([eles_a, eles_b])
    azs_a = np.linspace(0, az_range[1], nb // 4)
    azs_b = np.linspace(az_range[1], az_range[0], nb // 2)
    azs_c = np.linspace(az_range[0], 0, nb // 4)
    azs = np.concatenate([azs_a, azs_b, azs_c])

    for ele, az in zip(eles, azs):
        R, T = look_at_view_transform(radius, ele, az)
        cameras.append(FoVPerspectiveCameras(device=device, R=R, T=T, fov=FOV))

    return cameras


def render_obj(mesh, cameras, no_light=True, rez=512, aa_factor=2):
    raster_settings = RasterizationSettings(image_size=rez * aa_factor, blur_radius=0.0, faces_per_pixel=1)
    # lights = PointLights(device=device, location=cameras.get_camera_center()*-1)  # for IMesh
    lights = PointLights(device=device, location=cameras.get_camera_center())

    if not no_light:
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras,raster_settings=raster_settings),
            shader=SoftPhongShader(device=device,cameras=cameras,lights=lights)
        )
    else:
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras,raster_settings=raster_settings),
            shader=SimpleShader(device=device)
        )
    images = renderer(mesh)
    images = images.permute(0, 3, 1, 2)  # NHWC -> NCHW
    if aa_factor > 1:
        images = F.avg_pool2d(images, kernel_size=aa_factor, stride=aa_factor)
    images = images.permute(0, 2, 3, 1)  # NCHW -> NHWC
    images[..., 3] = images[..., 3] * 255
    images = images.clone().detach().cpu().to(torch.int).numpy()

    return  images

def plot_images(images):
    fig, axs = plt.subplots(3, 4, figsize=(20, 15))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(images[i][0, ..., :3])
        ax.axis('off')
    plt.show()


def render_mesh_test(mesh_file, save_folder):
        
    mesh_name = os.path.basename(mesh_file).split('.')[0]
    save_dir = os.path.join(save_folder, mesh_name)
    os.makedirs(save_dir, exist_ok=True)

    fov = math.degrees(0.8575560450553894)
    mesh = load_mesh(mesh_file, rotate='ours', normalize=True)
    # mesh = load_glb_mesh(mesh_file, normalize=True)
    cameras = get_render_cameras(fov, radius=3)
    for i, camera in enumerate(cameras):
        image = render_obj(mesh, camera, no_light=False)
        alpha = image[0, ..., 3:]
        image = image[0, ..., :3]
        alpha = np.clip(alpha * 2, 0, 255)
        image = np.concatenate([image, alpha], axis=-1)
        plt.imsave(os.path.join(save_dir, f'{i}.png'), image.astype('uint8'))

def render_mesh(mesh_file, save_folder):
        
    mesh_name = os.path.basename(mesh_file).split('.')[0]
    save_dir = os.path.join(save_folder, mesh_name)
    os.makedirs(save_dir, exist_ok=True)

    fov = math.degrees(0.8575560450553894)
    mesh = load_mesh(mesh_file, rotate='ours', normalize=True)
    # mesh = load_obj_mesh(mesh_file, rotate='ours', normalize=True)
    # mesh = load_glb_mesh(mesh_file, normalize=True)
    cameras_1 = get_render_cameras(fov, ele=20, nb=6, radius=3)
    cameras_2 = get_render_cameras(fov, ele=40, nb=6, radius=3)
    cameras_3 = get_render_cameras(fov, ele=60, nb=6, radius=3)
    cameras = cameras_1 + cameras_2 + cameras_3
    for i, camera in enumerate(cameras):
        image = render_obj(mesh, camera, no_light=False, rez=2048, aa_factor=1)
        alpha = image[0, ..., 3:]
        image = image[0, ..., :3]
        alpha = np.clip(alpha * 2, 0, 255)
        image = np.concatenate([image, alpha], axis=-1)
        plt.imsave(os.path.join(save_dir, f'render_{i:03d}.png'), image.astype('uint8'))

if __name__ == '__main__':

    parent_folder = '/data1/users/yuanhao/guying_proj/eval/graphdreamer_all'
    parent_save_folder = '/data1/users/yuanhao/guying_proj/eval/baselines_renders_2k/graphdreamer_all'
    names = os.listdir(parent_folder)
    for name in names:
        file = os.path.join(parent_folder, name, 'meshes', 'it10000-test-G.obj')
        if not os.path.exists(file):
            print(f'File {file} does not exist')
            continue
        save_folder = os.path.join(parent_save_folder, name)
        os.makedirs(save_folder, exist_ok=True)
        render_mesh(file, save_folder)

    # # parent_folder = '/data1/datasets/garment-data/all_objs/cloth4d/z0000'
    # parent_folder = '/data1/users/yuanhao/guying_proj/eval/baselines/midiroom4_optim_0'
    # parent_save_folder = '/data1/users/yuanhao/guying_proj/eval/baseline_renders/midiroom4_optim_0'
    # # select all ends with obj
    # names = glob.glob(os.path.join(parent_folder, '*.obj'))
    # names = [os.path.basename(name).split('.')[0] for name in names]
    # for name in names:
    #     file = os.path.join(parent_folder, f'{name}.obj')
    #     if not os.path.exists(file):
    #         print(f'File {file} does not exist')
    #         continue
    #     save_folder = os.path.join(parent_save_folder, name)
    #     os.makedirs(save_folder, exist_ok=True)
    #     render_mesh(file, save_folder)
