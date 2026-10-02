import os
import sys
import glob
import multiprocessing
import tqdm
import random
import threading
import torch


def render_multiview(source_folder, target_folder, gpu_id=0):
    # Get all the glb files in the source folder
    obj_filenames = glob.glob(os.path.join(source_folder, '*.glb')) + glob.glob(os.path.join(source_folder, '*.obj'))

    for obj_filename in tqdm.tqdm(obj_filenames):
        output_filename = os.path.join(target_folder, os.path.basename(obj_filename).split('.')[0])
        # Process each object
        torch.cuda.empty_cache()
        # first render a front view
        cmd = f'/data1/users/yuanhao/blender-4.0.0-linux-x64/blender --background --python ./blender_script.py -- --object_path {obj_filename}  --output_dir {output_filename}'
        cmd = f"export DISPLAY=:0.{gpu_id} && {cmd}"
        os.system(cmd)


def render_multiview_multiobjs(source_folder, target_folder, gpu_id=0):
    # Get all the glb files in the source folder
    obj_foldernames = glob.glob(os.path.join(source_folder, '*'))
    
    for obj_foldername in tqdm.tqdm(obj_foldernames):
        output_filename = os.path.join(target_folder, os.path.basename(obj_foldername).split('.')[0])
        # Process each object
        torch.cuda.empty_cache()
        # first render a front view
        # cmd = f'/data1/users/yuanhao/blender-4.0.0-linux-x64/blender --background --python ./blender_script.py -- --object_path {obj_filename}  --output_dir {output_filename}'
        cmd = f'/data1/users/yuanhao/blender-4.0.0-linux-x64/blender --background --python ./blender_script_multi_objs.py -- --object_path {obj_foldername}  --output_dir {output_filename}'
        cmd = f"export DISPLAY=:0.{gpu_id} && {cmd}"
        os.system(cmd)


if __name__ == "__main__":
    gpu_id = 2

    # source_parent_folder = '/data1/users/yuanhao/guying_proj/eval/baselines'
    # target_parent_folder = '/data1/users/yuanhao/guying_proj/eval/baselines_renders_2k'
    # # target_folder = '/data1/datasets/garment-data/image-data/test'

    # baseline_names = os.listdir(source_parent_folder)
    # # baseline_names = ['midiroom4_optim_0']
    # for name in baseline_names:
    #     source_folder = os.path.join(source_parent_folder, name)
    #     target_folder = os.path.join(target_parent_folder, name)
    #     os.makedirs(target_folder, exist_ok=True)
    #     # render_multiview
    #     render_multiview(source_folder, target_folder, gpu_id=gpu_id)

    source_parent_folder = '/data1/users/yuanhao/guying_proj/eval/ours'
    target_parent_folder = '/data1/users/yuanhao/guying_proj/eval/ours/last1_renders'
    # target_folder = '/data1/datasets/garment-data/image-data/test'

    # baseline_names = os.listdir(source_parent_folder)
    baseline_names = ['last1']
    for name in baseline_names:
        source_folder = os.path.join(source_parent_folder, name)
        target_folder = os.path.join(target_parent_folder, name)
        os.makedirs(target_folder, exist_ok=True)
        # render_multiview
        render_multiview_multiobjs(source_folder, target_folder, gpu_id=gpu_id)

    
    