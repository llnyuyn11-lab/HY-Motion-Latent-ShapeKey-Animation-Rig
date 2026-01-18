# demos/latent_shapekey/hy_adapter.py
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
except Exception:
    torch = None


@dataclass
class MotionResult:
    # minimal "portable" container
    data: Any                  # tensor/np/dict, whatever HY-Motion gives you
    path: str                  # saved file path on disk
    meta: Dict[str, Any]       # prompt/seed/steps/length/etc.


class HYMotionAdapter:
    def __init__(self, device: str = "cuda", out_dir: str = "outputs_latent_shapekey"):
        self.device = device
        here = os.path.dirname(os.path.abspath(__file__))
        self.out_dir = os.path.join(here, out_dir)
        os.makedirs(self.out_dir, exist_ok=True)


        # TODO (later): load HY-Motion model/runtime here
        # e.g. self.model = ...
        # e.g. self.pipeline = ...

    # ---------- helpers ----------
    def _to_numpy(self, x: Any):
        """Convert torch tensors (or nested structures) to numpy safely."""
        if torch is not None and isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy()
        return x

    def _make_run_id(self) -> str:
        return time.strftime("%Y%m%d_%H%M%S")

    def _save_npz(self, payload: Dict[str, Any], out_path: str):
        """
        Save a dict of stuff into .npz.
        - tensors become numpy
        - dict/list objects are stored as object arrays (still readable in python)
        """
        converted = {}
        for k, v in payload.items():
            v2 = self._to_numpy(v)
            # allow dicts/lists to be saved too (object array)
            if isinstance(v2, (dict, list, tuple)):
                converted[k] = np.array(v2, dtype=object)
            else:
                converted[k] = v2
        np.savez_compressed(out_path, **converted)

    # ---------- main API ----------
    def sample_motion(
        self,
        prompt: str,
        z,
        seed: int,
        steps: int,
        length: int,
        key_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        1) (now) returns a placeholder 'motion' but also SAVES an .npz artifact.
        2) (later) replace `motion = ...` with real HY-Motion inference output,
           and keep the same saving logic.
        """
        run_id = self._make_run_id()
        out_path = os.path.join(self.out_dir, f"motion_{run_id}_seed{seed}.npz")

        run_id = self._make_run_id()
        out_path = os.path.join(self.out_dir, f"motion_{run_id}_seed{seed}.npz")
        out_path = os.path.abspath(out_path)   # ← ADD THIS LINE


        # -----------------------------
        # TODO LATER: replace this block with real HY-Motion generation.
        # For now we just store latent and settings so you can compare A/B/C runs.
        # -----------------------------
        motion = {
            "type": "placeholder",
            "note": "Replace HYMotionAdapter.sample_motion() with real HY-Motion inference.",
            "latent": z,
        }

        meta = {
            "prompt": prompt,
            "seed": int(seed),
            "steps": int(steps),
            "length": int(length),
            "device": self.device,
            "key_weights": key_weights or {},
            "latent_shape": getattr(z, "shape", None),
        }

        # Save EVERYTHING you might want to inspect later
        payload = {
            "motion": motion,
            "latent": z,
            "meta": meta,
        }
        self._save_npz(payload, out_path)

        # Return a JSON-friendly dict for Gradio
        return {
            "status": "OK (saved .npz)",
            "saved_path": out_path,
            "meta": meta,
        }
    
        LEFT_ARM  = {6, 7, 8}
        RIGHT_ARM = {10, 11, 12}
        SPINE     = {1, 2, 3, 4}
        LEFT_LEG  = {13, 14, 15, 16}
        RIGHT_LEG = {17, 18, 19, 20}
  
        





    def export_preview(self, motion: Any, out_path: str = "preview.bvh", wA=1.0, wB=0.5, wC=0.3):

        """
        BVH Export (full human-ish skeleton):

        - Root has 6 channels: Xposition Yposition Zposition Zrotation Xrotation Yrotation
        - All other joints have 3 channels: Zrotation Xrotation Yrotation
        - Motion data supported:
            A) motion can be dict with:
                - motion["root_pos"] : (T,3) in same units as offsets (e.g., cm)
                - motion["euler_deg"]: (T,J,3) degrees in ZXY channel order per joint index
            B) otherwise: exports a static pose (still valid BVH)
        """
        import os
        import numpy as np

        out_path = os.path.abspath(out_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # -------------------------
        # 1) Skeleton definition
        # -------------------------
        # Each joint: (name, parent_index, offset_xyz)
        # Units: arbitrary (Blender will scale); these are "human-ish" proportions.
        SKELETON = [
            ("Hips",       -1, (0.0, 0.0, 0.0)),     # 0  ROOT
            ("Spine",       0, (0.0, 10.0, 0.0)),    # 1
            ("Spine1",      1, (0.0, 10.0, 0.0)),    # 2
            ("Neck",        2, (0.0, 10.0, 0.0)),    # 3
            ("Head",        3, (0.0,  8.0, 0.0)),    # 4

            ("LeftShoulder",2, (5.0,  9.0, 0.0)),    # 5
            ("LeftArm",     5, (12.0, 0.0, 0.0)),    # 6
            ("LeftForeArm", 6, (12.0, 0.0, 0.0)),    # 7
            ("LeftHand",    7, (10.0, 0.0, 0.0)),    # 8

            ("RightShoulder",2, (-5.0,  9.0, 0.0)),  # 9
            ("RightArm",     9, (-12.0, 0.0, 0.0)),  # 10
            ("RightForeArm",10, (-12.0, 0.0, 0.0)),  # 11
            ("RightHand",   11, (-10.0, 0.0, 0.0)),  # 12

            ("LeftUpLeg",    0, ( 6.0, -10.0, 0.0)), # 13
            ("LeftLeg",     13, ( 0.0, -18.0, 0.0)), # 14
            ("LeftFoot",    14, ( 0.0, -18.0, 4.0)), # 15
            ("LeftToe",     15, ( 0.0,   0.0, 8.0)), # 16

            ("RightUpLeg",   0, (-6.0, -10.0, 0.0)), # 17
            ("RightLeg",    17, ( 0.0, -18.0, 0.0)), # 18
            ("RightFoot",   18, ( 0.0, -18.0, 4.0)), # 19
            ("RightToe",    19, ( 0.0,   0.0, 8.0)), # 20
        ]
        J = len(SKELETON)

        # Build child lists
        children = [[] for _ in range(J)]
        for i, (_, p, _) in enumerate(SKELETON):
            if p >= 0:
                children[p].append(i)

        # -------------------------
        # 2) Read motion if present
        # -------------------------
        # Default: 120 frames of static pose
        T = 120
        root_pos = None
        euler_deg = None
        latent = None


        if isinstance(motion, dict):
            # Length hint (optional)
            meta = motion.get("meta", {}) if isinstance(motion.get("meta", {}), dict) else {}
            T = int(meta.get("length", motion.get("length", T)))

            latent = motion.get("latent", None)

            if latent is not None:
                # --- make latent always CPU numpy (handles torch cuda tensors) ---
                try:
                    import torch
                    if isinstance(latent, torch.Tensor):
                        latent = latent.detach().float().cpu().numpy()
                except Exception:
                    pass

                latent = np.asarray(latent, dtype=np.float32)

                
            if "root_pos" in motion:
                root_pos = np.asarray(motion["root_pos"], dtype=np.float32)  # (T,3)
                T = int(root_pos.shape[0])

            # Prefer explicit keys if present
            if "euler_deg" in motion:
                euler_deg = np.asarray(motion["euler_deg"], dtype=np.float32)
                T = int(euler_deg.shape[0])

            # Fallback: some saves store rotations under motion["motion"]["euler_deg"] etc.
            elif "motion" in motion and isinstance(motion["motion"], dict) and "euler_deg" in motion["motion"]:
                euler_deg = np.asarray(motion["motion"]["euler_deg"], dtype=np.float32)
                T = int(euler_deg.shape[0])

            # Latent: also allow nested
            if "latent" in motion:
                # --- FIX: accept torch tensor on GPU/CPU ---
                x = motion["latent"]
                if hasattr(x, "detach"):  # torch tensor
                    x = x.detach()
                    if hasattr(x, "cpu"):
                        x = x.cpu()
                    x = x.numpy()
                latent = np.asarray(x, dtype=np.float32)

            elif "motion" in motion and isinstance(motion["motion"], dict) and "latent" in motion["motion"]:
                latent = np.asarray(motion["motion"]["latent"], dtype=np.float32)



        # Safe fallbacks
        if root_pos is None:
            root_pos = np.zeros((T, 3), dtype=np.float32)
        if euler_deg is None:
            euler_deg = np.zeros((T, J, 3), dtype=np.float32)

        # Clamp if someone passes weird shapes
        if euler_deg.shape[1] != J:
            # If joint count mismatch, keep it valid but static
            euler_deg = np.zeros((T, J, 3), dtype=np.float32)

        # -------------------------
        # 3) BVH writing helpers
        # -------------------------
        def write_joint(f, idx, indent, is_root=False):
            name, parent, (ox, oy, oz) = SKELETON[idx]
            tab = "  " * indent

            if is_root:
                f.write(f"{tab}ROOT {name}\n")
            else:
                f.write(f"{tab}JOINT {name}\n")

            f.write(f"{tab}{{\n")
            f.write(f"{tab}  OFFSET {ox:.6f} {oy:.6f} {oz:.6f}\n")

            if is_root:
                f.write(f"{tab}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")
            else:
                f.write(f"{tab}  CHANNELS 3 Zrotation Xrotation Yrotation\n")

            if len(children[idx]) == 0:
                # End Site
                f.write(f"{tab}  End Site\n")
                f.write(f"{tab}  {{\n")
                f.write(f"{tab}    OFFSET 0.000000 5.000000 0.000000\n")
                f.write(f"{tab}  }}\n")
            else:
                for c in children[idx]:
                    write_joint(f, c, indent + 1, is_root=False)

            f.write(f"{tab}}}\n")

        # -------------------------
        # 4) Write BVH file
        # -------------------------
        fps = 30.0
        frame_time = 1.0 / fps

        with open(out_path, "w", newline="\n") as f:
            f.write("HIERARCHY\n")
            write_joint(f, 0, indent=0, is_root=True)

            f.write("MOTION\n")
            f.write(f"Frames: {T}\n")
            f.write(f"Frame Time: {frame_time:.6f}\n")

            # BVH motion row order must match hierarchy traversal order.
            # Our traversal order is: ROOT then depth-first children in the order we wrote them.
            # We'll rebuild the traversal index list so motion lines match exactly.
            traversal = []
            def dfs(i):
                traversal.append(i)
                for c in children[i]:
                    dfs(c)
            dfs(0)

            # Write each frame row:
            # Root: pos + rot, Others: rot only
            import math

            LEFT_ARM  = {6, 7, 8}      # shoulder, elbow, wrist
            RIGHT_ARM = {10, 11, 12}
            SPINE     = {1, 2, 3, 4}
            LEFT_LEG  = {13, 14, 15, 16}
            RIGHT_LEG = {17, 18, 19, 20}

            latent = None
            if isinstance(motion, dict):
                latent = motion.get("latent", None)

                if latent is not None:
                    # latent might be a torch.Tensor on GPU; move to CPU before numpy
                    try:
                        import torch  # only used for type check
                        if isinstance(latent, torch.Tensor):
                            latent = latent.detach().cpu().numpy()
                        else:
                            latent = np.asarray(latent)
                    except Exception:
                        # if torch isn't available for some reason, fall back to numpy
                        latent = np.asarray(latent)


            def L(dim: int, t: int) -> float:
                if latent is None:
                    return 0.0
                # latent could be (1,512) or (T,512)
                if latent.ndim == 2 and latent.shape[0] == T:
                    return float(latent[t, dim])
                return float(latent[0, dim])
            
            # --- Read controls (shape-key weights) from motion dict ---

            controls = {}
            if isinstance(motion, dict):
                controls = motion.get("controls", {}) or {}

            wA = float(controls.get("wA", 1.0))
            wB = float(controls.get("wB", 0.5))
            wC = float(controls.get("wC", 0.3))



            for t in range(T):
                parts = []

                # Root position (XYZ)
                px, py, pz = root_pos[t].tolist()
                parts += [f"{px:.6f}", f"{py:.6f}", f"{pz:.6f}"]

                # Root rotation (Z X Y)
                rz0, rx0, ry0 = euler_deg[t, 0].tolist()
                parts += [f"{rz0:.6f}", f"{rx0:.6f}", f"{ry0:.6f}"]



                # Optional: add a time wave so you can SEE it clearly
                wave = math.sin(2.0 * math.pi * t / T)


                

               
                for j in traversal[1:]:
                    rz, rx, ry = euler_deg[t, j].tolist()

                           
                     # === SHAPE-KEY STYLE OFFSETS (edit these) ===
                    # Use latent dims as weights (or replace with sliders later)
                    # ---- per-joint offsets (shape-key rig) ----
                    a = L(0, t) * wA
                    b = L(1, t) * wB
                    c = L(2, t) * wC

                    if j in LEFT_ARM:
                        rx += (35.0 * a) * wave
                        rz += (10.0 * c) * wave

                    if j in RIGHT_ARM:
                        rx += (35.0 * a) * wave
                        rz -= (10.0 * c) * wave

                    if j in SPINE:
                        ry += (20.0 * b) * wave

                    if j in LEFT_LEG:
                        rx += (15.0 * a) * wave

                    if j in RIGHT_LEG:
                        rx -= (15.0 * a) * wave
                    # ------------------------------------------

                   

                    parts += [f"{rz:.6f}", f"{rx:.6f}", f"{ry:.6f}"]

                expected = 6 + 3 * (len(traversal) - 1)
                if len(parts) != expected:
                    raise RuntimeError(f"Frame {t}: got {len(parts)} values, expected {expected}")

                f.write(" ".join(parts) + "\n")


            return out_path

