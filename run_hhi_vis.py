"""
visualize_filtered_hhi.py

HHI BVH Motion Replay in Newton.
Synchronized with 'generate_hhi_mjcf.py':
1. Uses embedded BVH parser (Hierarchy + Motion).
2. Filters joint data to match the filtered XML skeleton.
3. Maps raw BVH names to standardized XML names.
"""

import sys
import os
import numpy as np
import warnings
import warp as wp
import time

import argparse
import math
from scipy.spatial.transform import Rotation as R
from PIL import Image
import hhi_utils

# Suppress gimbal lock warnings
warnings.filterwarnings("ignore", message="Gimbal lock detected")

# --- Newton Path Configuration ---
# Ensure Newton is in your python path
sys.path.append("/media/rh/codes/sim/newton")
try:
    import newton
except ImportError:
    print("Error: 'newton' module not found. Please check sys.path.")
    sys.exit(1)


# --- Main Replayer Class ---

class HHIReplay:
    def __init__(self, viewer, bvh_paths, scene_xml, static=False, rest_mode=False):
        self.viewer = viewer
        self.static = static
        self.rest_mode = rest_mode
        self.scene_xml = scene_xml
        self.anims = {}
        self.bvh_items = []

        # Load BVHs
        xml_name = os.path.basename(scene_xml)
        for prefix, path in bvh_paths.items():
            if "AA.xml" in xml_name and prefix != "cg": continue
            if "RM.xml" in xml_name and prefix != "cr": continue
            
            if os.path.exists(path):
                # If rest mode, we only need hierarchy, not motion (faster)
                # NOTE: We load motion even if static, so we can jump to a specific frame.
                self.anims[prefix] = hhi_utils.read_bvh(path, read_motion=not getattr(self, 'rest_mode', False))
                self.bvh_items.append((prefix, path))
            
        # Build Model
        print("Initializing Newton Model...")
        # Build Model
        print("Initializing Newton Model...")
        builder = newton.ModelBuilder()
        
        # Helper to load includes manually (Newton doesn't support <include> well)
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(scene_xml)
            root = tree.getroot()
            base_dir = os.path.dirname(os.path.abspath(scene_xml))
            
            # Load includes first
            for inc in root.findall('include'):
                inc_file = inc.get('file')
                if inc_file:
                    inc_path = os.path.join(base_dir, inc_file)
                    if os.path.exists(inc_path):
                        print(f"  [Newton] Loading included XML: {inc_file}")
                        builder.add_mjcf(inc_path)
                    else:
                        print(f"  [Warning] Included file not found: {inc_path}")
                        
            # Then load the scene itself (for worldbody, etc.)
            builder.add_mjcf(scene_xml)
            
        except Exception as e:
            print(f"Error parsing scene XML includes: {e}")
            # Fallback
            builder.add_mjcf(scene_xml)

        self.model = builder.finalize()
        
        if self.viewer:
            self.viewer.set_model(self.model)
        
        self.state = self.model.state()
        
        # Determine FPS and Num Frames from BVH data
        if not self.bvh_items:
            raise ValueError("No BVH files loaded. Cannot determine frame rate.")
        
        first_anim = self.anims[self.bvh_items[0][0]]
        
        if first_anim.positions is None:
            if self.rest_mode:
                # Rest mode doesn't need motion data
                self.num_frames = 1
                self.fps = 30.0  # Arbitrary for rest mode, only used for viewer timing
                print("Rest mode: No motion data needed.")
            else:
                raise ValueError("No motion data in BVH file and not in rest mode.")
        else:
            self.num_frames = first_anim.positions.shape[0]
            if first_anim.frametime <= 0:
                raise ValueError(f"Invalid frame time in BVH: {first_anim.frametime}")
            self.fps = 1.0 / first_anim.frametime
            print(f"Set FPS to {self.fps:.2f} based on BVH Frame Time {first_anim.frametime:.6f}")
        
        self.sim_time = 0.0
        self.update_pose(0)

    def update_pose(self, frame_idx):
        if self.rest_mode:
            return # Keep at rest pose (0)
            
        if not self.bvh_items: return
            
        qpos_vals = []
        
        # Coordinate transformation: BVH (Y-up) -> MJCF (Z-up)
        # T_ROOT maps: X->X, Y->Z, Z->-Y
        T = hhi_utils.T_ROOT
        
        for prefix, _ in self.bvh_items:
            anim = self.anims[prefix]
            fi = frame_idx % self.num_frames
            
            # --- PART 1: ROOT TRANSFORMATION (Global) ---
            # Handles BVH channels 0-5 (Pos + Rot) -> MJCF Freejoint (Pos + Quat)
            
            # 1. Position: Apply T_ROOT to convert (x,y,z) from Y-up to Z-up
            root_pos_bvh = anim.positions[fi]
            root_pos_mjcf = T @ root_pos_bvh
            qpos_vals.extend(root_pos_mjcf.tolist())
            
            # 2. Orientation: Convert Euler(Y-up) -> Quat(Z-up)
            # BVH Euler (in Y-up) -> Matrix R_local
            # Transform Operator: R_global = T_ROOT * R_local * T_ROOT_inv
            order = anim.orders[0] if anim.orders[0] else "zxy"
            r_vals = [anim.rotations[fi, 0, {'x':0, 'y':1, 'z':2}[char]] for char in order]
            
            # Use Intrinsic rotations (Uppercase) for BVH
            R_local = R.from_euler(order.upper(), r_vals, degrees=True)
            R_trans = R.from_matrix(T)
            R_global = R_trans * R_local * R_trans.inv()
            
            # Newton/Scipy use (x, y, z, w) order. Confirmed.
            root_quat_mjcf = R_global.as_quat() 
            qpos_vals.extend(root_quat_mjcf.tolist())
            
            # --- PART 2: JOINT LOCAL ROTATIONS (Local) ---
            # Iterate all other bones (Index 1..). 
            # Appends hinge joint angles in radians.
            
            for j in range(1, len(anim.bones)):
                raw_name = anim.bones[j].split(':')[-1]
                
                info = hhi_utils.get_bone_info(raw_name)
                
                # Only process if bone is mapped AND has a joint prefix defined
                if info and info['joint']:
                    order = anim.orders[j] or "zxy"
                    
                    for char in order:
                        val_idx = {'x':0, 'y':1, 'z':2}[char]
                        deg = anim.rotations[fi, j, val_idx]
                        rad = np.radians(deg)
                        qpos_vals.append(rad)

        q_np = np.array(qpos_vals, dtype=np.float32)
        
        # Safety Check
        if len(q_np) != self.model.joint_q.shape[0]:
            print(f"Frame {frame_idx}: Mismatch! Data {len(q_np)} vs Model {self.model.joint_q.shape[0]}")
            # print("  Likely cause: BONE_MAPPING in Replayer doesn't match Generator.")
            return
        
        self.state.joint_q = wp.from_numpy(q_np, dtype=wp.float32, device=self.state.joint_q.device)
        self.state.joint_qd.zero_()
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

    def step(self):
        if not self.static:
            self.sim_time += 1.0 / self.fps # Advance exactly one BVH frame
            frame = int(self.sim_time * self.fps)
            self.update_pose(frame)

    def render(self):
        if self.viewer:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state)
            self.viewer.end_frame()



def plot_motion_data(anims, output_dir, start_frame=2):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] Matplotlib not found, skipping plots.")
        return

    print(f"[INFO] Generating motion plots in {output_dir}...")

    for prefix, anim in anims.items():
        # --- Plot 1: Base (Root) Position and Euler Angles ---
        frames = np.arange(anim.rotations.shape[0])
        
        # Transform base position: BVH Y-up -> MJCF Z-up
        # T_ROOT maps: X->X, Y->Z, Z->-Y
        T = hhi_utils.T_ROOT
        root_pos_bvh = anim.positions  # [Frames, 3] in BVH Y-up
        root_pos_mjcf = (T @ root_pos_bvh.T).T  # [Frames, 3] in MJCF Z-up
        
        # Transform base Euler angles to Z-up quaternion, then back to Euler for plotting
        # This matches what the visualizer does
        order = anim.orders[0] if anim.orders[0] else "zxy"
        root_euler_mjcf = np.zeros_like(anim.rotations[:, 0, :])
        
        r_trans = R.from_matrix(T)
        for fi in range(len(frames)):
            # Get rotation values in BVH order
            r_vals = []
            for char in order:
                idx = {'x': 0, 'y': 1, 'z': 2}[char]
                r_vals.append(anim.rotations[fi, 0, idx])
            
            # Convert to quaternion, transform to Z-up
            # Use Intrinsic rotations (Uppercase) for BVH
            r_local = R.from_euler(order.upper(), r_vals, degrees=True)
            r_global = r_trans * r_local * r_trans.inv()
            
            # Convert back to Euler XYZ for plotting
            root_euler_mjcf[fi] = r_global.as_euler('xyz', degrees=True)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Top: Position (in meters, MJCF Z-up coordinates)
        axes[0].plot(frames, root_pos_mjcf[:, 0], label='X')
        axes[0].plot(frames, root_pos_mjcf[:, 1], label='Y')
        axes[0].plot(frames, root_pos_mjcf[:, 2], label='Z (up)')
        axes[0].set_title(f'{prefix.upper()} Base Position (MJCF Z-up, meters) - GLOBAL')
        axes[0].set_ylabel('Position (m)')
        axes[0].axvline(x=start_frame, color='red', linestyle='--', alpha=0.7)
        axes[0].legend()
        axes[0].grid(True)
        
        # Bottom: Euler Angles (transformed to Z-up, XYZ order for display)
        axes[1].plot(frames, root_euler_mjcf[:, 0], label='X')
        axes[1].plot(frames, root_euler_mjcf[:, 1], label='Y')
        axes[1].plot(frames, root_euler_mjcf[:, 2], label='Z')
        axes[1].set_title(f'{prefix.upper()} Base Euler Angles (MJCF Z-up, XYZ) - GLOBAL')
        axes[1].set_xlabel('Frame')
        axes[1].set_ylabel('Angle (deg)')
        axes[1].axvline(x=start_frame, color='red', linestyle='--', alpha=0.7, label=f'Start (Frame {start_frame})')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        base_plot_path = os.path.join(output_dir, f'{prefix}_base_euler.png')
        plt.savefig(base_plot_path)
        plt.close()
        print(f" Saved {base_plot_path}")

        # --- Plot 2: All Other Joints ---
        # There are many joints, so we use subplots
        num_joints = len(anim.bones) - 1
        if num_joints > 0:
            cols = 4
            rows = (num_joints + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
            axes = axes.flatten()
            
            for j in range(1, len(anim.bones)):
                ax = axes[j-1]
                joint_name = anim.bones[j]
                data = anim.rotations[:, j, :]
                
                ax.plot(frames, data[:, 0], label='X', alpha=0.7)
                ax.plot(frames, data[:, 1], label='Y', alpha=0.7)
                ax.plot(frames, data[:, 2], label='Z', alpha=0.7)
                ax.set_title(f'{joint_name}')
                ax.grid(True)
                ax.axvline(x=start_frame, color='red', linestyle='--', alpha=0.7)
                if j == 1: ax.legend() # Legend only on first to save space
            
            # Hide empty subplots
            for k in range(num_joints, len(axes)):
                axes[k].axis('off')
                
            plt.suptitle(f'{prefix.upper()} Joint Euler Angles - LOCAL (degrees)', fontsize=14, y=1.0)
            plt.tight_layout()
            joints_plot_path = os.path.join(output_dir, f'{prefix}_joints_euler.png')
            plt.savefig(joints_plot_path)
            plt.close()
            print(f" Saved {joints_plot_path}")
    
    print("[SUCCESS] Plots generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--start_frame", type=int, default=2, help="Frame to start visualization/plotting marker from")
    parser.add_argument("--rest", action="store_true", help="Visualize rest pose (no motion data)")
    parser.add_argument("--config", type=str, default="hhi_config.yaml", help="Path to YAML config file")
    # Removed explicit xml_file argument since it's now derived from config name
    args, unknown = parser.parse_known_args()
    
    # Load Config to get Dataset Name
    try:
        bvh_files = hhi_utils.load_config(args.config)
        dataset_name = bvh_files.get("name", "default")
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Infer XML Path
    xml_filename = f"hhi_scene_{dataset_name}.xml"
    
    # Path Setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(base_dir, xml_filename)
        
    if not os.path.exists(xml_path):
        print(f"XML not found: {xml_path}")
        print(f"Did you run 'generate_hhi_mjcf.py --config {args.config}'?")
        sys.exit(1)

    # Base dir for output
    base_dir = os.path.dirname(os.path.abspath(xml_path))

    # Viewer Setup
    viewer = None
    if not args.headless:
        try:
            viewer = newton.viewer.ViewerGL()
        except:
            print("Viewer init failed, using headless.")
            args.headless = True
    else:
        # Explicit headless request (e.g. for generating preview on server)
        import pyglet
        pyglet.options['headless'] = True
        viewer = newton.viewer.ViewerGL(headless=True)



    sim = HHIReplay(viewer, bvh_files, xml_path, static=args.static, rest_mode=args.rest)
    
    # Generate Plots (Only if not in rest mode)
    if not args.rest:
        plot_motion_data(sim.anims, base_dir, start_frame=args.start_frame)

    if args.headless:
        if args.static:
            # Headless Static Preview Generation
            
            # Advance to desired frame
            sim.update_pose(args.start_frame)
            
            # Set Camera (Dynamic)
            # Match logic from interactive mode (Face: -3.5 Y)
            anim_key = 'cg' if 'cg' in sim.anims else ('cr' if 'cr' in sim.anims else None)
            
            if anim_key:
                 anim = sim.anims[anim_key]
                 fi = args.start_frame % anim.num_frames
                 raw = anim.positions[fi]
                 t_np = hhi_utils.T_ROOT @ raw
                 
                 cam_x = float(t_np[0])
                 cam_y = float(t_np[1]) - 4.5
                 cam_z = float(t_np[2]) + 1.0
                 
                 viewer.set_camera(wp.vec3(cam_x, cam_y, cam_z), -5.0, 90.0)
            else:
                 viewer.set_camera(wp.vec3(0.0, -4.5, 1.0), -5.0, 90.0)

            # Settle & Render
            for i in range(10):
                viewer.begin_frame(time=i*0.016)
                viewer.log_state(sim.state)
                viewer.end_frame()

            # Save
            img_wp = viewer.get_frame()
            img_np = img_wp.numpy()
            
            preview_path = os.path.join(base_dir, "hhi_scene_preview.png")
            Image.fromarray(img_np).save(preview_path)
            print(f"[SUCCESS] Saved preview to {preview_path} (Frame {args.start_frame})")
        else:
            # Just run blindly
            for i in range(100): sim.step()
    else:
        # Interactive
        
        # Set Camera (Match static preview)
        if viewer:
             # Dynamically center camera on the CG character
             anim_key = 'cg' if 'cg' in sim.anims else ('cr' if 'cr' in sim.anims else None)
             
             if anim_key and sim.anims[anim_key].num_frames > 0 and sim.anims[anim_key].positions is not None:
                 anim = sim.anims[anim_key]
                 fi = args.start_frame % anim.num_frames
                 raw_pos = anim.positions[fi]
                 
                 # Target (Root) in current frame
                 t_np = hhi_utils.T_ROOT @ raw_pos
                 
                 # Camera Position Strategy:
                 # Character faces -Y (Newton). To see face, place camera at Y-3.5 relative to character.
                 # Height Z+1.5 to look down slightly.
                 cam_x = float(t_np[0])
                 cam_y = float(t_np[1]) - 4.5
                 cam_z = float(t_np[2]) + 1.0
                 
                 # Pitch -5 (look slightly down), Yaw 90 (look +Y)
                 viewer.set_camera(wp.vec3(cam_x, cam_y, cam_z), -5.0, 90.0)
                 
                 print(f"[Viewer] Camera positioned at ({cam_x:.1f}, {cam_y:.1f}, {cam_z:.1f}) facing {anim_key}")
             else:
                 # Fallback (Rest Mode or no motion data)
                 # Assume root is at origin
                 viewer.set_camera(wp.vec3(0.0, -4.5, 1.0), -5.0, 90.0)

        # If start frame is set, jump there?
        sim.sim_time = args.start_frame / sim.fps
        sim.update_pose(args.start_frame)
        
        # Throttle to Native BVH FPS 
        target_dt = 1.0 / sim.fps
        last_time = time.time()
        
        while viewer.is_running():
            now = time.time()
            if now - last_time >= target_dt:
                sim.step()
                sim.render()
                last_time = now
            else:
                time.sleep(0.001)