import os
import glob
from typing import Optional, Tuple

import numpy as np
import gradio as gr

from hy_adapter import HYMotionAdapter

LIVE_BVH_PATH = os.path.join("outputs_latent_shapekey", "live.bvh")


def guess_default_npz() -> str:
    # pick the newest motion_*.npz in outputs_latent_shapekey/
    candidates = sorted(
        glob.glob(os.path.join("outputs_latent_shapekey", "motion_*.npz")),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    return candidates[0] if candidates else ""


def export_bvh_from_npz(npz_path: str, wA: float, wB: float, wC: float) -> str:
    npz_path = npz_path.strip().strip('"').strip("'")
    if not npz_path:
        raise ValueError("NPZ path is empty.")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    d = np.load(npz_path, allow_pickle=True)

    meta = {}
    if "meta" in d.files:
        meta_arr = d["meta"]
        meta = meta_arr.item() if getattr(meta_arr, "dtype", None) == object else {}

    if "latent" not in d.files:
        raise KeyError(f"'latent' not found in npz. keys={d.files}")

    motion = {
        "meta": meta,
        "latent": d["latent"],
        # IMPORTANT: controls live INSIDE the motion dict
        "controls": {"wA": float(wA), "wB": float(wB), "wC": float(wC)},
    }

    out_dir = "outputs_latent_shapekey"
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(npz_path))[0]
    os.makedirs("outputs_latent_shapekey", exist_ok=True)
    out_path = LIVE_BVH_PATH


    HYMotionAdapter().export_preview(motion, out_path)
    return os.path.abspath(out_path)


def ui_generate(npz_path: str, wA: float, wB: float, wC: float) -> Tuple[str, Optional[str]]:
    try:
        bvh_path = export_bvh_from_npz(npz_path, wA, wB, wC)
        msg = (
            "✅ BVH written:\n"
            f"{bvh_path}\n\n"
            "In Blender: File → Import → Motion Capture (.bvh)\n"
            "Tip: import into a NEW armature each time (or delete old action)."
        )
        return msg, bvh_path
    except Exception as e:
        return f"❌ {type(e).__name__}: {e}", None


def main():
    default_npz = guess_default_npz()

    with gr.Blocks(title="HY-Motion Latent Shape-Key Controls") as demo:
        gr.Markdown(
            """
# Motion Shape-Key Rig (Latent → BVH)

Move the sliders and click **Generate BVH**.
This writes a new BVH using your `hy_adapter.py` mixing logic.

**Important:** `hy_adapter.py` must read `motion["controls"]` inside `export_preview()`.
"""
        )

        npz_path = gr.Textbox(
            label="NPZ path (motion_*.npz)",
            value=default_npz,
            placeholder=r"outputs_latent_shapekey\motion_YYYYMMDD_HHMMSS_seed123.npz",
        )

        with gr.Row():
            wA = gr.Slider(-3.0, 3.0, value=1.0, step=0.01, label="wA (shape key A)")
            wB = gr.Slider(-3.0, 3.0, value=0.5, step=0.01, label="wB (shape key B)")
            wC = gr.Slider(-3.0, 3.0, value=0.3, step=0.01, label="wC (shape key C)")

        btn = gr.Button("Generate BVH", variant="primary")
        status = gr.Textbox(label="Status", lines=6)
        out_file = gr.File(label="BVH output")

        btn.click(fn=ui_generate, inputs=[npz_path, wA, wB, wC], outputs=[status, out_file])

    demo.launch()


if __name__ == "__main__":
    main()
