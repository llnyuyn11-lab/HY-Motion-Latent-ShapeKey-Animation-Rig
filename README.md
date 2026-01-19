A Blender-friendly “motion shape-key rig” that maps latent controls to joint-space offsets, controlled via Gradio sliders and exported as BVH for instant preview.


(A) How to run the UI
Example (Windows):

cd HY-Motion-Latent-ShapeKey-Rig
pip install -r requirements.txt
python gradio_controls.py



(B) How to use in Blender

import BVH once

run blender_reload_live_bvh.py to reload latest live.bvh



(C) Behind-the-scene Diagram


## Latent Shape-Key Rig — Technical Overview

This project introduces a **Shape-Key–style latent control system** for generative motion models, designed to support **artist-friendly iteration** and **closed-loop integration with 3D software (Blender)**.

Selected latent dimensions are exposed as continuous controls, enabling real-time performance sculpting without re-running inference.

---

## Technical Roadmap

![Technical roadmap](BTS Docs/roadmap.png)

This roadmap outlines the evolution of the system from a modular foundation to a fully closed-loop animation workflow, prioritizing usability, iteration speed, and production ergonomics.

---

## Phase 1 — Folder Structure & System Architecture

![Phase 1](BTS Docs/Phase1.png)

Phase 1 establishes a clean, modular architecture that can live alongside existing AI motion models (e.g. HY-Motion) or operate independently.

**Core components:**
- **`demo_gradio.py`**  
  Handles UI layout, Gradio wiring, and user interaction logic.
- **`HY_adapter.py`**  
  Bridges the latent rig to the HY-Motion model and handles motion export (NPZ / BVH).
- **`latent_bank.py`**  
  Encapsulates latent math operations and shape-key–style blending logic.

The structure separates UI, model adaptation, and latent math to ensure maintainability across future model updates.

---

## Phase 2 — Adjusting the Gradio UI Page

![Phase 2](BTS Docs/Phase2.png)

Phase 2 improves usability for real animation workflows after correctness and output validation.

**Key adjustments:**
1. **Full Human Rig Hierarchy Export**  
   Exports a complete joint hierarchy with channels, eliminating manual BVH retargeting in Blender.
2. **Expanded Latent Range Controls**  
   Slider ranges extended from `[0, 1]` to `[-1, 2]` with clearer distance indicators for expressive exploration.
3. **Customizable Save Folder**  
   Users can select output directories to match studio file organization.
4. **Clear Output Notifications**  
   Explicit display of the saved BVH file path after export.

---

## Phase 3 — 3D Software UX “Closed-Loop” (Blender)

![Phase 3](BTS Docs/Phase3.png)

Phase 3 closes the iteration loop between AI generation and DCC tools, with Blender as the target environment.

**Closed-loop upgrades:**
1. **One-Click BVH Refresh (Gradio)**  
   Gradio overwrites a consistent output file (`live.bvh`) instead of generating new files per iteration.
2. **One-Button BVH Reload (Blender)**  
   A lightweight Blender script enables instant reloading of the latest BVH, behaving like a native Blender feature.

This removes file-management friction and enables continuous iteration.

---

## Closed-Loop Animation Workflow

![Closed loop workflow](BTS Docs/Close loop logic.png)

**Iteration loop:**
1. Run HY-Motion inference → generate latent motion (NPZ)  
2. Adjust latent “shape-key” sliders in Gradio → write `live.bvh`  
3. In Blender, click **Reload BVH** → view updated motion instantly  
4. Repeat without re-running inference

This enables rapid performance sculpting driven by intuitive controls rather than execution cost.

---

## System Architecture & Responsibility Separation

![System architecture](BTS Docs/System Architecture.png)

The system is deliberately decoupled into three layers:
- **Inference Layer** — HY-Motion generates latent representations (NPZ)
- **Control Layer** — Latent shape-key rig exposes continuous controls
- **DCC Integration Layer** — Blender handles visualization via one-click reload

This separation allows real-time iteration without retraining or re-executing the generative model.

---

## Latent → Shape-Key Mapping

![Latent mapping](BTS Docs/Latent → Shape-Key Mapping.png)

Selected latent dimensions are mapped to animator-friendly sliders (WA / WB / WC), functioning like traditional shape keys.

Each slider produces structured, joint-aware motion offsets, enabling intuitive control over performance style using familiar animation metaphors.

---

## Key Design Decisions & Tradeoffs

![Design tradeoffs](BTS Docs/Key Design Decisions and Tradeoffs.png)

- **Exchange format:** BVH  
  Simple, universal, lightweight, and DCC-friendly.
- **Control interface:** Gradio-based shape-key rig  
  Decoupled from DCC UI for faster iteration.
- **Reload strategy:** One-click overwrite + reload  
  Optimizes iteration speed and reduces cognitive overhead.

The system prioritizes iteration speed, interoperability, and animator ergonomics over tightly coupled or model-specific solutions.
