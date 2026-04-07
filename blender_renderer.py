"""Blender Renderer Module - Using bpy directly like Spoken2Sign"""
import os
import sys


class BlenderRenderer:
    """Render sign language animations using Blender Python API (bpy)"""
    
    def __init__(self, blender_path=None, render_script_path=None):
        """
        Initialize Blender Renderer
        
        Args:
            blender_path: Not used (kept for compatibility)
            render_script_path: Path to render script directory
        """
        self.render_script_path = render_script_path or './scripts'
        
        # Try to import bpy
        try:
            import bpy
            self.bpy = bpy
            self.bpy_available = True
            print(" Blender Python API (bpy) available")
        except ImportError:
            self.bpy_available = False
            print(" Blender Python API (bpy) not available")
            print("  Install with: pip install bpy==3.4.0")
    
    def render(self, motion_dir, output_dir, video_id='output'):
        """
        Render motion files to images using Blender Python API
        
        Args:
            motion_dir: Directory containing motion PKL files
            output_dir: Directory to save rendered images
            video_id: Video identifier
            
        Returns:
            num_images: Number of images rendered
        """
        if not self.bpy_available:
            raise RuntimeError(
                "Blender Python API (bpy) not available.\n"
                "Install with: pip install bpy==3.4.0"
            )
        
        import bpy
        import pickle
        from pathlib import Path
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Initializing Blender for rendering...")
        
        # Get paths relative to script directory
        script_dir = Path(__file__).parent
        pretrained_dir = script_dir / 'pretrained_models'
        
        # Load addon & blend file
        addon_path = str(pretrained_dir / 'smplx_blender_addon_300_20220623.zip')
        blend_file = str(pretrained_dir / 'smplx_ronglai.blend')
        
        if not os.path.exists(addon_path):
            raise FileNotFoundError(f"SMPL-X addon not found: {addon_path}")
        if not os.path.exists(blend_file):
            raise FileNotFoundError(f"Blend file not found: {blend_file}")
        
        print(f"  Loading addon: {addon_path}")
        bpy.ops.preferences.addon_install(filepath=addon_path, overwrite=True)
        bpy.ops.preferences.addon_enable(module='smplx_blender_addon')
        bpy.ops.wm.save_userpref()
        
        print(f"  Loading blend file: {blend_file}")
        bpy.ops.wm.open_mainfile(filepath=blend_file)
        
        addon_data_path = str(pretrained_dir / 'smplx_blender_addon' / 'data')
        bpy.ops.file.find_missing_files(directory=addon_data_path)
        
        # Setup render settings
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
        
        # Get motion files
        motion_files = sorted([f for f in os.listdir(motion_dir) if f.endswith('.pkl')])
        
        if not motion_files:
            raise ValueError(f"No motion files found in {motion_dir}")
        
        print(f"  Rendering {len(motion_files)} frames...")
        
        # Render each frame
        for i, motion_file in enumerate(motion_files):
            motion_path = os.path.join(motion_dir, motion_file)
            
            try:
                # Deselect all
                bpy.ops.object.select_all(action='DESELECT')
                
                # Find SMPL-X armature - GIỐNG FILE GỐC
                armature_obj = None
                for obj in bpy.context.scene.objects:
                    if 'smpl' in obj.name.lower() and obj.type == 'ARMATURE':
                        armature_obj = obj
                        break
                
                if armature_obj:
                    armature_obj.select_set(True)
                    bpy.context.view_layer.objects.active = armature_obj
                    bpy.ops.object.smplx_load_pose(filepath=motion_path)
                else:
                    print(f"   Warning: No SMPL-X Armature object found.")
                    continue
                        
            except Exception as e:
                print(f"   Error loading pose {motion_path}: {e}")
                continue
                
            # Render
            bpy.ops.render.render()
            
            # Save image
            output_file = os.path.join(output_dir, f"{i:04d}.png")
            bpy.data.images["Render Result"].save_render(output_file)
            # Bật dòng dưới nếu bạn muốn thấy log lưu file
            # print(f" Saved: {output_file}")
            
            if (i + 1) % 10 == 0:
                print(f"  Rendered {i+1}/{len(motion_files)} frames")
        
        # Count rendered images
        num_images = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
        print(f" Rendered {num_images} images to {output_dir}")
        
        return num_images