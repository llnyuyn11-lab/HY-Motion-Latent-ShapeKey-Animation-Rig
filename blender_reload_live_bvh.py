# blender_reload_live_bvh.py
# Blender: Scripting tab -> Text Editor -> Open this file -> Run Script
# Each run deletes the previous imported rig and re-imports the latest live.bvh

import bpy
import os

# --- CHANGE THIS IF NEEDED ---
BVH_PATH = r"C:\Users\llnyu\Desktop\HY-Motion-1.0\demos\latent_shapekey\outputs_latent_shapekey\live.bvh"
RIG_NAME = "LIVE_BVH_RIG"

def delete_previous_rig(name: str):
    # Delete armature object
    obj = bpy.data.objects.get(name)
    if obj:
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.delete(use_global=False)

    # Delete armature data blocks that may remain
    arm = bpy.data.armatures.get(name)
    if arm:
        bpy.data.armatures.remove(arm)

def import_bvh(path: str, rig_name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"BVH not found: {path}")

    # Import BVH
    bpy.ops.import_anim.bvh(
        filepath=path,
        filter_glob="*.bvh",
        global_scale=1.0,
        frame_start=1,
        use_fps_scale=True,
        update_scene_fps=True,
        rotate_mode='NATIVE',
    )

    # Blender imports an armature and makes it active
    imported_obj = bpy.context.view_layer.objects.active
    if imported_obj is None or imported_obj.type != "ARMATURE":
        # Try to find the newest armature
        armatures = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
        imported_obj = armatures[-1] if armatures else None

    if imported_obj:
        imported_obj.name = rig_name
        if imported_obj.data:
            imported_obj.data.name = rig_name

def main():
    delete_previous_rig(RIG_NAME)
    import_bvh(BVH_PATH, RIG_NAME)

    # Make sure timeline shows the action
    bpy.context.scene.frame_set(1)
    print("Reloaded BVH:", BVH_PATH)

main()
