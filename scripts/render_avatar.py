import bpy
import numpy as np
from math import radians
# from mathutils import Matrix
import pickle
import os, argparse
from cmd_parser import parse_config
from tqdm import tqdm
import random; random.seed(0)
import math


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args, remaining = parser.parse_known_args()
    args = parse_config(remaining)

    # --- Load addon & file blend gốc ---
    bpy.ops.preferences.addon_install(filepath='pretrained_models/smplx_blender_addon_300_20220623.zip', overwrite=True)
    bpy.ops.preferences.addon_enable(module='smplx_blender_addon')
    bpy.ops.wm.save_userpref()
    bpy.ops.wm.open_mainfile(filepath="pretrained_models/smplx_ronglai.blend")
    
    path = os.path.abspath('pretrained_models/smplx_blender_addon/data')
    bpy.ops.file.find_missing_files(directory=path)

    # --- Thiết lập render ---
    scene = bpy.data.scenes['Scene']
    scene.render.resolution_y = 512
    scene.render.resolution_x = 512
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    # --- SỬA ĐỔI: Áp dụng vị trí camera và bỏ đèn từ code cũ ---
    cam = bpy.data.objects["Camera"]
    # Vị trí camera từ code cũ
    cam.location = (cam.location.x, -1, 0.155) 
    
    # Bỏ góc quay tường minh để Blender tự tính toán, giống code cũ
    # cam.rotation_euler = ... # Dòng này được bỏ đi
    
    # Xóa đèn "CameraLight" nếu nó tồn tại, để giống với thiết lập của code cũ
    if "CameraLight" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["CameraLight"], do_unlink=True)
    # -----------------------------------------------------------------

    # --- Load danh sách video ---
    with open('data/idx_vn.pkl', 'rb') as f:
        video_ids = pickle.load(f)
    init_idx = args['init_idx']
    num_per_proc = args['num_per_proc']
    start_idx = init_idx
    end_idx = start_idx + num_per_proc
    video_ids = video_ids[start_idx:end_idx]

    # --- Vòng lặp render ---
    for video_id in tqdm(video_ids):
        motion_path = os.path.join('./motions', video_id)
        if not os.path.exists(motion_path):
            continue
        motion_lst = sorted(os.listdir(motion_path))
        
        img_dir = os.path.join('./images', video_id)
        os.makedirs(img_dir, exist_ok=True)

        for i, motion_file in enumerate(motion_lst):
            fname = os.path.join(motion_path, motion_file)
            
            try:
                bpy.ops.object.select_all(action='DESELECT')
                armature_obj = None
                for obj in bpy.context.scene.objects:
                    if 'smpl' in obj.name.lower() and obj.type == 'ARMATURE':
                        armature_obj = obj
                        break
                
                if armature_obj:
                    armature_obj.select_set(True)
                    bpy.context.view_layer.objects.active = armature_obj
                    bpy.ops.object.smplx_load_pose(filepath=fname)
                else:
                    print(f" Warning: No SMPL-X Armature object found.")
                    continue
                        
            except Exception as e:
                print(f" Error loading pose {fname}: {e}")
                continue
                
            bpy.ops.render.render()
            out_file = os.path.join(img_dir, f"images{i:04d}.png")
            bpy.data.images["Render Result"].save_render(out_file)
            # Bật dòng dưới nếu bạn muốn thấy log lưu file
            # print(f" Saved: {out_file}")