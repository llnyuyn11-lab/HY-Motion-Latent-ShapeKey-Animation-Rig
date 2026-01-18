A Blender-friendly “motion shape-key rig” that maps latent controls to joint-space offsets, controlled via Gradio sliders and exported as BVH for instant preview.


(A) How to run the UI
Example (Windows):

cd HY-Motion-Latent-ShapeKey-Rig
pip install -r requirements.txt
python gradio_controls.py



(B) How to use in Blender

import BVH once

run blender_reload_live_bvh.py to reload latest live.bvh