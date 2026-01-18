import numpy as np
from hy_adapter import HYMotionAdapter

npz_path = "outputs_latent_shapekey/motion_20260114_062458_seed123.npz"
d = np.load(npz_path, allow_pickle=True)

motion = {
    "meta": (d["meta"].item() if d["meta"].dtype == object else {}),
    "latent": d["latent"],
}

a = HYMotionAdapter()

# Export three variants (like turning shape keys on one-by-one)
for name, controls in [
    ("CTRL_wA2", {"wA": 2.0, "wB": 0.0, "wC": 0.0}),
    ("CTRL_wB2", {"wA": 0.0, "wB": 2.0, "wC": 0.0}),
    ("CTRL_wC2", {"wA": 0.0, "wB": 0.0, "wC": 2.0}),
]:
    motion["controls"] = controls
    a.export_preview(motion, f"outputs_latent_shapekey/{name}.bvh")
    print("WROTE", name)
