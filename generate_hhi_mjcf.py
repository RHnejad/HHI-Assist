import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import hhi_utils
import argparse

# =============================================================================
# BODY CLASS DEFINITION
# =============================================================================

class Body:
    """
    Represents a body/bone in the skeleton.
    Encapsulates hierarchy, geometry, and physics data.
    """
    def __init__(self, name: str, mapping_info: dict, role: str):
        self.name = name                 # Clean name (e.g. "Spine")
        self.role = role                 # "cg" or "cr"
        
        # From Mapping
        self.link_name = mapping_info.get('link')
        self.joint_name = mapping_info.get('joint')
        self.joint_prefix = mapping_info.get('joint')
        self.parent_name = mapping_info.get('parent')
        self.children_names = mapping_info.get('children', [])
        
        # Graph Pointers (Filled later)
        self.parent_obj = None           # Pointer to Parent Body object
        self.children_objs = []          # List of Child Body objects
        self.endsite_obj = None          # Pointer to EndSite Body object (if exists)
        
        # From BVH
        self.raw_name = ""               # Full BVH name (e.g. "Hips")
        self.offset = np.zeros(3)        # Offset from Parent to Me (Local)
        self.rotation_order = "zxy"      # Euler order
        self.bvh_idx = -1                # NEW: Store BVH index for sorting
        
        # Geometry Attributes
        self.radius = 0.04
        self.color = "0.5 0.5 0.5 1"
        self.is_endsite = False
        self.is_root = (self.parent_name is None)

    def set_bvh_data(self, idx, raw_name, offset, order):
        # Transform BVH (Y-up) vector to MJCF (Z-up)
        # We store the *Transformed* offset in the object
        self.offset = hhi_utils.T_ROOT @ np.array(offset)
        self.rotation_order = order.lower() if order else None
        self.raw_name = raw_name
        self.bvh_idx = idx
        
        # Update Geometry Props
        self.radius = hhi_utils.get_radius(self.name)
        self.color = hhi_utils.get_color(self.role, self.name)
        
        # Refine Root Color/Size
        if self.is_root:
             self.color = hhi_utils.get_color(self.role, 'Hips') # Ensure Hips color

    def get_mass(self, density=1000):
        """Calculate mass based on geometry volume * density."""
        if self.is_endsite:
            # EndSites have their own logic in EndSiteBody
            return 0.0 
        
        # Geometry is Capsule from 0,0,0 to self.offset (My Length)
        length = np.linalg.norm(self.offset)
        if length < 0.001 and not self.is_root:
            return 0.0
            
        radius = self.radius
        
        if self.is_root:
            # Sphere
            vol = (4/3) * np.pi * (radius**3)
        else:
            # Capsule: Cylinder + 2 Half-Spheres (Total = Cyl + Sphere Volume)
            # Volume = pi * r^2 * h + (4/3) * pi * r^3
            vol = (np.pi * (radius**2) * length) + ((4/3) * np.pi * (radius**3))
            
        return vol * density

class EndSiteBody(Body):
    """Specialized Body for EndSites (Tips)."""
    def __init__(self, parent_obj):
        # Fake mapping info
        super().__init__(f"EndSite_{parent_obj.name}", {}, parent_obj.role)
        self.is_endsite = True
        self.parent_obj = parent_obj
        # Parent Name logic handled by pointers, but for consistency:
        self.parent_name = parent_obj.name
        
        # Naming Convention: "endsite-{parent_name}"
        self.name = f"endsite-{parent_obj.name}" 
        self.link_name = f"{parent_obj.link_name}_end" # e.g. HeadL_end
        
        # Color inherits from parent
        self.color = parent_obj.color
        self.radius = 0.02 # Default
        
        self.raw_name = None
        self.offset = np.zeros(3)
        self.rotation_order = None
        self.bvh_idx = -1  # NEW: Store BVH index for sorting
        
        self.shape_type = 'capsule' # Default
        self.length = 0.0
        
    def set_bvh_data(self, idx, raw_name, offset, order):
         # Similar to Body, but no rotation usually
         self.bvh_idx = idx
         self.raw_name = raw_name
         self.offset = hhi_utils.T_ROOT @ offset  # Y-up to Z-up
         self.rotation_order = order.lower() if order else None
         self.length = np.linalg.norm(self.offset)

    def get_mass(self, density=1000):
        # EndSite Geometry depends on type
        if self.shape_type == 'sphere':
             vol = (4/3) * np.pi * (self.radius**3)
        else:
             # Capsule length is self.length? No, self.offset!
             length = np.linalg.norm(self.offset)
             vol = (np.pi * (self.radius**2) * length) + ((4/3) * np.pi * (self.radius**3))
        return vol * density

# =============================================================================
# GENERATOR CLASS
# =============================================================================

class HHIModelGenerator:
    def __init__(self, bvh_path, role, subject_id):
        self.bvh_path = bvh_path
        self.role = role
        self.subject_id = subject_id
        
        # Storage
        self.bodies = {} # name -> Body object
        self.root_body = None
        
        # Helpers
        print(f"  Parsing BVH: {os.path.basename(bvh_path)}")
        self.parser = hhi_utils.BVHParser(bvh_path)
        self.parser.parse(read_motion=False)

    def calculate_total_mass(self):
        """Calculates total mass of the generated character."""
        total_mass = 0.0
        # Iterate all bodies
        stack = [self.root_body]
        while stack:
            body = stack.pop()
            total_mass += body.get_mass()
            stack.extend(body.children_objs)
            
            # Check EndSite
            if body.endsite_obj:
                total_mass += body.endsite_obj.get_mass()
                
        return total_mass

    def print_statistics(self):
        print(f"  [Statistics] Character {self.subject_id}:")
        total_mass = self.calculate_total_mass()
        print(f"    Total Mass: {total_mass:.2f} kg")
        print(f"    Link Lengths:")
        
        # traversal
        stack = [self.root_body]
        while stack:
            body = stack.pop(0) # BFS
            
            if not body.is_root:
                length = np.linalg.norm(body.offset)
                print(f"      - {body.link_name:<20}: {length*100:.2f} cm")
            
            if body.endsite_obj:
                es = body.endsite_obj
                length = np.linalg.norm(es.offset)
                print(f"      - {es.link_name:<20}: {length*100:.2f} cm")
                
            stack.extend(body.children_objs)

    def build(self):
        print(f"  Building Object Graph for {self.subject_id}...")
        self._create_bodies_from_mapping()
        self._link_hierarchy()
        # 1. Fill BVH Data (Offsets, Orders)
        self._fill_bvh_data()
        
        # 2. Sort Children to match BVH Order (Critical for Visualization Indexing)
        self._sort_children(self.root_body)
        
        # 3. Generate XML
        print(f"  Generating XML...")
        mjcf_root = self._generate_xml_tree()
        return mjcf_root


    def _sort_children(self, body):
        """Recursively sort children bodies by their BVH index."""
        # Sort children based on the bvh_idx we stored
        # If bvh_idx is -1 (shouldn't happen for mapped bones), put at end
        body.children_objs.sort(key=lambda x: x.bvh_idx if x.bvh_idx != -1 else 9999)
        
        for child in body.children_objs:
            self._sort_children(child)
            
    def _create_bodies_from_mapping(self):
        """Step 1: Create Body objects from BONE_MAPPING keys."""
        for name, info in hhi_utils.BONE_MAPPING.items():
            body = Body(name, info, self.role)
            self.bodies[name] = body
            
            if body.parent_name is None:
                self.root_body = body

    def _link_hierarchy(self):
        """Step 2: Link parent/child pointers."""
        for name, body in self.bodies.items():
            # Link Parent
            if body.parent_name:
                parent = self.bodies.get(body.parent_name)
                if parent:
                    body.parent_obj = parent
                    parent.children_objs.append(body)
                else:
                    print(f"Warning: Parent {body.parent_name} not found for {name}")

    def _fill_bvh_data(self):
        """Step 3: Parse BVH and fill offsets."""
        # We iterate the BVH bones.
        # BVH defines Structure. MAPPING defines Logic.
        # We match BVH Bone Name -> Clean Name -> Body Object.
        
        print(f"    - Filling data from {len(self.parser.bones)} BVH bones")
        
        for i, raw_name in enumerate(self.parser.bones):
            clean_name = raw_name.split(':')[-1]
            
            if clean_name == "End Site":
                # End Site Logic
                # Find parent index in BVH
                p_idx = self.parser.parents[i]
                p_raw = self.parser.bones[p_idx]
                p_clean = p_raw.split(':')[-1]
                
                parent_body = self.bodies.get(p_clean)
                if parent_body:
                    # Create EndSite Object attached to Parent
                    es = EndSiteBody(parent_body)
                    es.set_bvh_data(i, es.name, self.parser.offsets[i], None)
                    
                    # Refine EndSite Geometry
                    if "Head" in p_clean:
                        es.shape_type = "sphere"
                        es.radius = es.length * 0.5 # Head Sphere Logic
                    else:
                        es.shape_type = "capsule"
                        
                    parent_body.endsite_obj = es
            
            else:
                # Regular Bone
                body = self.bodies.get(clean_name)
                if body:
                    body.set_bvh_data(i, raw_name, self.parser.offsets[i], self.parser.orders[i])
                else:
                    # Bone in BVH but not in Mapping? (Shouldn't happen for core skeleton)
                    pass

    def _fmt(self, vec):
        return f"{vec[0]:.6f} {vec[1]:.6f} {vec[2]:.6f}"
    
    def _fmt_axis(self, vec):
        return f"{vec[0]:.0f} {vec[1]:.0f} {vec[2]:.0f}"

    def _generate_xml_tree(self):
        """Step 4: Traverse graph and build ET."""
        root = ET.Element('mujoco', model=f"hhi_{self.subject_id}")
        
        # Standard Imports
        ET.SubElement(root, 'compiler', angle='radian', meshdir='assets')
        default = ET.SubElement(root, 'default')
        # Add basic visual defaults if needed
        
        worldbody = ET.SubElement(root, 'worldbody')
        
        # Start recursion from Root
        if self.root_body:
            self._build_body_xml(self.root_body, worldbody)
            
        return root

    def _build_body_xml(self, body, parent_xml):
        # NEW LOGIC: Body Position = Parent's Offset
        # Geometry = My Offset (Length)
        
        # 1. Determine Position (Parent's Offset)
        if body.is_root:
            pos_str = "0 0 0"
        else:
            # We need the PARENT's offset.
            # body.parent_obj should exist.
            if body.parent_obj:
                # Parent's offset is stored in parent_obj.offset
                pos_str = self._fmt(body.parent_obj.offset)
            else:
                # Fallback (Shouldn't happen for non-root)
                pos_str = "0 0 0"

        # XML Body Name
        body_xml_name = body.raw_name if body.raw_name else body.name
        
        # 2. Create Body Element
        # Note: Geoms are now INSIDE the Body (User Request + Logic Consistency)
        body_elem = ET.SubElement(parent_xml, 'body', name=body_xml_name, pos=pos_str)
        
        # 3. Geometry (My Length)
        if body.is_root:
            # Root: Simple Sphere at origin
            ET.SubElement(body_elem, 'geom', 
                          name=body.link_name, type='sphere', 
                          size=f"{body.radius:.4f}", rgba=body.color)
        else:
            # Child Body
            # Drawn from 0 0 0 (Start of Segment, my Joint location) 
            # to self.offset (End of Segment)
            geom_len = np.linalg.norm(body.offset)
            if geom_len > 0.0001:
                ET.SubElement(body_elem, 'geom',
                              name=body.link_name,
                              type='capsule',
                              fromto=f"0 0 0 {self._fmt(body.offset)}",
                              size=f"{body.radius:.4f}",
                              rgba=body.color)

        # 4. Joint (Rotates Me)
        # Joints are always inside the Body they rotate
        if body.is_root:
            ET.SubElement(body_elem, 'freejoint', name=f"{body.link_name}_free")
        elif body.joint_name:
            # Create Hinge Joints
            base_axes = {'x': [1,0,0], 'y': [0,1,0], 'z': [0,0,1]}
            order = body.rotation_order if body.rotation_order else "zxy"
            
            for char in order:
                axis_name = f"{body.joint_prefix}_{char}"
                ax_vec = hhi_utils.T_ROOT @ np.array(base_axes[char])
                ET.SubElement(body_elem, 'joint',
                              name=axis_name,
                              type='hinge',
                              axis=self._fmt_axis(ax_vec),
                              range="-180 180")

        # 5. Handle EndSite (If attached to Me)
        if body.endsite_obj:
            es = body.endsite_obj
            # EndSite Body Position = My Offset (End of My Segment)
            # This matches the pattern: ES Pos = Parent(Me) Offset
            es_pos_str = self._fmt(body.offset)
            
            es_body_elem = ET.SubElement(body_elem, 'body', name=es.name, pos=es_pos_str)
            
            # EndSite Geometry (Inside ES Body)
            # From 0 0 0 -> ES Offset
            if es.shape_type == 'sphere':
                 mid = es.offset * 0.5
                 ET.SubElement(es_body_elem, 'geom',
                               name=es.link_name,
                               type='sphere',
                               pos=self._fmt(mid),
                               size=f"{es.radius:.4f}",
                               rgba=es.color)
            else:
                 ET.SubElement(es_body_elem, 'geom',
                               name=es.link_name,
                               type='capsule',
                               fromto=f"0 0 0 {self._fmt(es.offset)}",
                               size=f"{es.radius:.4f}",
                               rgba=es.color)

        # 6. Handle Children (Recurse)
        for child in body.children_objs:
            self._build_body_xml(child, body_elem)


# =============================================================================
# MAIN
# =============================================================================

def generate_body_from_bvh(bvh_path, role):
    filename = os.path.basename(bvh_path)
    # Extract ID: ..._AA_50.bvh -> AA
    parts = filename.split('_')
    if len(parts) >= 2:
        subject_id = parts[-2]
    else:
        subject_id = f"Sub{role.upper()}"
    
    gen = HHIModelGenerator(bvh_path, role, subject_id)
    xml_tree = gen.build()
    
    # Save
    out_name = f"robot_{role}_{subject_id}.xml"
    
    # Format
    xml_str = ET.tostring(xml_tree, encoding='unicode')
    reparsed = minidom.parseString(xml_str)
    pretty = reparsed.toprettyxml(indent="  ")
    
    with open(out_name, 'w') as f:
        f.write(pretty)
    print(f"  Generated: {os.path.abspath(out_name)}")
    
    # NEW: Print Statistics (Mass + Lengths)
    gen.print_statistics()
    
    return out_name, subject_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="hhi_config.yaml")
    args = parser.parse_args()
    
    print(f"Output directory: {os.getcwd()}")
    
    try:
        bvh_files = hhi_utils.load_config(args.config)
        
        # Generate Characters
        cg_xml, cg_id = generate_body_from_bvh(bvh_files['cg'], "cg")
        cr_xml, cr_id = generate_body_from_bvh(bvh_files['cr'], "cr")
        
        # Generate Scene
        scene_name = f"hhi_scene_{bvh_files['name']}.xml"
        
        root = ET.Element('mujoco', model=f"Scene_{bvh_files['name']}")
        ET.SubElement(root, 'include', file=cg_xml)
        ET.SubElement(root, 'include', file=cr_xml)
        
        # Add Floor/Light/Skybox
        world = ET.SubElement(root, 'worldbody')
        ET.SubElement(world, 'light', diffuse=".5 .5 .5", pos="0 0 3", dir="0 0 -1")
        ET.SubElement(world, 'geom', name="floor", type="plane", size="0 0 0.05", material="grid") # simplified
        
        tree = ET.ElementTree(root)
        tree.write(scene_name)
        print(f"  Generated: {os.path.abspath(scene_name)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
