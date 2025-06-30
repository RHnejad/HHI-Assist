import numpy as np
from scipy.spatial.transform import Rotation

LOCAL = False

class Joint:
    def __init__(self, name, offset, parent=None):
        self.name = name # Name of the joint
        self.offset = np.array(offset, dtype=float) # offset position of the joint relative to to its parent
        self.parent = parent # parent joint
        self.children = [] # list of child joints
        self.channels = [] # channels for motion data
        self.motion_index = None  # Index where this joint's motion data starts

    def add_child(self, child):
        self.children.append(child)
        
def euler_to_rot_matrix(euler_angles):
    # Convert Euler angles ( in degrees ) to a rotation matrix 

    rz, rx, ry = np.radians(euler_angles)
    cz = np.cos(rz)
    sz = np.sin(rz)
    cy = np.cos(ry)
    sy = np.sin(ry)
    cx = np.cos(rx)
    sx = np.sin(rx)

    Rz = np.array([[cz, -sz, 0],
                   [sz, cz, 0],
                   [0, 0, 1]])
    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]])

    return Rz @ Rx @ Ry
    
class BVHParser:
    def __init__(self, file_path):
        self.file_path = file_path # Path to the BVH file
        self.root = None # root joint of the skeleton
        self.joints = {} # Dictionnary of joints by name
        self.frames = 0 # Number of frames in the motion data
        self.frame_time = 0.0 # Time per frame
        self.motion_data = [] # List to store motion data
        self.total_channels = 0  # Total number of channels

    def parse(self):
        with open(self.file_path, 'r') as file:
            lines = iter(file.readlines())

        current_joint = None
        joint_stack = []
        mode = 'HIERARCHY'
        for line in lines:
            tokens = line.strip().split()
            if not tokens:
                continue

            if tokens[0] in ['ROOT', 'JOINT']:
                name = tokens[1]
                parent = current_joint
                current_joint = Joint(name, offset=[0.0, 0.0, 0.0], parent=parent)
                if parent:
                    parent.add_child(current_joint)
                self.joints[name] = current_joint
                if tokens[0] == 'ROOT':
                    self.root = current_joint
                joint_stack.append(current_joint)
            elif tokens[0] == 'OFFSET':
                current_joint.offset = np.array(list(map(float, tokens[1:])))
            elif tokens[0] == 'CHANNELS':
                n_channels = int(tokens[1])
                current_joint.channels = tokens[2:]
                current_joint.motion_index = self.total_channels
                self.total_channels += n_channels
            elif tokens[0] == 'End':
                end_site = Joint(name='End Site', offset=[0.0, 0.0, 0.0], parent=current_joint)
                current_joint.add_child(end_site)
                while True:
                    line = next(lines).strip()
                    if line == '}':
                        break
            elif tokens[0] == '}':
                if joint_stack:
                    joint_stack.pop()
                    current_joint = joint_stack[-1] if joint_stack else None
            elif tokens[0] == 'MOTION':
                mode = 'MOTION'
            elif mode == 'MOTION':
                if tokens[0] == 'Frames:':
                    self.frames = int(tokens[1])
                elif tokens[0].startswith('Frame'):
                    self.frame_time = float(tokens[2])
                else:
                    self.motion_data.append(np.array(list(map(float, tokens))))
        self.motion_data = np.array(self.motion_data)

    def compute_world_positions(self):
        def compute_joint_world_position(joint, frame_data):
            if joint.parent:
                parent_transform = compute_joint_world_position(joint.parent, frame_data)
                rotation = euler_to_rot_matrix(frame_data[joint.motion_index:joint.motion_index+3])
                
            else:
                parent_transform = np.eye(4)
                rotation = euler_to_rot_matrix(frame_data[joint.motion_index+3:joint.motion_index+6])
                joint.offset = frame_data[joint.motion_index:joint.motion_index+3]
            
            joint_transform = np.eye(4)
            joint_transform[:3, :3] = rotation
            joint_translate = np.eye(4)
            joint_translate[:3, 3] = joint.offset
            #print("joint translate")
            #print(joint_translate)
            #print("joint transform")
            #print(joint_transform)
            #print("joint transform @ parent_transform")
            #print(joint_transform @ parent_transform)
            #print("parent_transform @ joint_translate @ joint_transform")
            #print(parent_transform @ joint_translate @ joint_transform)
            return parent_transform @ joint_translate @ joint_transform

        world_positions = np.zeros((self.frames, self.total_channels // 3 - 1, 3))
        for frame_index in range(self.frames):
            frame_data = self.motion_data[frame_index]
            for joint_name, joint in self.joints.items():
                #print()
                #print(joint_name)
                #print()
                if joint.motion_index is not None:
                    #joint_pos = compute_joint_world_position(joint, frame_data)
                    joint_pos = compute_joint_world_position(joint, frame_data)[:3, 3]
                    if joint.parent:
                        world_positions[frame_index, joint.motion_index // 3 - 1] = joint_pos
                    else:
                        world_positions[frame_index, joint.motion_index // 3] = joint_pos
        return world_positions
    
    def get_rotations(self):
        return self.motion_data[:,3:]
    
    def get_rotations_normed(self):
        return self.motion_data[:,3:]/180
    
    def quat_from_euler_xyz(self, roll, pitch, yaw):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)

        qw = cy * cr * cp + sy * sr * sp
        qx = cy * sr * cp - sy * cr * sp
        qy = cy * cr * sp + sy * sr * cp
        qz = sy * cr * cp - cy * sr * sp

        quat = np.stack((qx, qy, qz, qw), axis=-1)#.squeeze(1)
        return quat
    
    def absolut_to_relative(self, global_quats): #undoing the absolute euler angles to relative euler angles
        #parent child relations: 
        p1 = [0, 13, 14, 15, 16] # 0 -> 13 -> 14 -> 15 -> 16 #left leg
        p2 = [0, 17, 18, 19, 20] # 0 -> 17 -> 18 -> 19 -> 20 #right leg
        p3 = [0, 1, 2, 3, 4] # 0 -> 1 -> 2 -> 3 -> 4  #head
        p4 = [0, 1, 2, 5, 6, 7, 8] # 0 -> 1 -> 2 -> 5 -> 6 -> 7 -> 8 #left arm
        p5 = [0, 1, 2, 9, 10, 11, 12] # 0 -> 1 -> 2 -> 9 -> 10 -> 11 -> 12 #right arm
        
        # p1 = [0, 5, 6, 7, 8] # 0 -> 13 -> 14 -> 15 -> 16 #left leg
        # p2 = [0, 1, 2, 3, 4] # 0 -> 17 -> 18 -> 19 -> 20 #right leg
        # p3 = [0, 9, 10, 19, 20] # 0 -> 1 -> 2 -> 3 -> 4  #head
        # p4 = [0, 9, 10, 15, 16, 17, 18] # 0 -> 1 -> 2 -> 5 -> 6 -> 7 -> 8 #left arm
        # p5 = [0, 9, 10, 11, 12, 13, 14] # 0 -> 1 -> 2 -> 9 -> 10 -> 11 -> 12 #right arm
        
        P = [p1, p2, p3, p4, p5]
        
        global_rotations = {}
        
        for i in range(global_quats.shape[1]): #global_quats: (24,21,4)
            global_rotations[i] = Rotation.from_quat(global_quats[:,i])
        
        local_euler_angles = {}
        local_euler_angles[0] = global_rotations[i].as_euler('ZXY', degrees=True)
        
        for p in P:
            for i in range(1, len(p)):
                parent_idx = p[i-1]
                child_idx = p[i]
                
                parent_rotation = global_rotations[parent_idx]
                child_rotation = global_rotations[child_idx]
                relative_rotation = parent_rotation.inv() * child_rotation if not LOCAL else child_rotation #(p_g^-1)*(p_g*ch_l) #Danger:child_rotation 
                 
                if child_idx not in local_euler_angles:
                    local_euler_angles[child_idx] = relative_rotation.as_euler('ZXY', degrees=True)
        
        # euler_angles_array = np.zeros((global_quats.shape[0], 3 * len(local_euler_angles)))
        euler_angles_array = np.zeros((global_quats.shape[0], len(local_euler_angles), 3))
        for key in sorted(local_euler_angles.keys()):
            # start_idx = key * 3
            # euler_angles_array[:, start_idx:start_idx+3] = local_euler_angles[key] 
            euler_angles_array[:, key, :] = local_euler_angles[key] 

        return euler_angles_array
                

    def absolute_quat(self, euler_angles): #new 
        
        #parent child relations: 
        p1 = [0, 13, 14, 15, 16] # 0 -> 13 -> 14 -> 15 -> 16 #left leg
        p2 = [0, 17, 18, 19, 20] # 0 -> 17 -> 18 -> 19 -> 20 #right leg
        p3 = [0, 1, 2, 3, 4] # 0 -> 1 -> 2 -> 3 -> 4  #head
        p4 = [0, 1, 2, 5, 6, 7, 8] # 0 -> 1 -> 2 -> 5 -> 6 -> 7 -> 8 #left arm
        p5 = [0, 1, 2, 9, 10, 11, 12] # 0 -> 1 -> 2 -> 9 -> 10 -> 11 -> 12 #right arm
        
        P = [p1, p2, p3, p4, p5]
        
        absolute_euler_angles = {}
        
        absolute_euler_angles[0] = Rotation.from_euler('ZXY', euler_angles[:,0:3].copy(), degrees=True)
                
        #adding each parents angle to the children 
        for p in P:
            for i in range(1, len(p)):
                child = Rotation.from_euler('ZXY', euler_angles[:,p[i]*3:p[i]*3+3], degrees=True)
                parent = absolute_euler_angles[p[i-1]]
                
                if p[i] not in absolute_euler_angles:
                    absolute_euler_angles[p[i]] =  parent * child if not LOCAL else child #Danger: child #
                    
        return  np.array([absolute_euler_angles[key].as_quat() for key in sorted(absolute_euler_angles.keys())]).transpose(1,0,2)
    
    def get_quaternions(self):
        euler_angles_rad = self.get_rotations()
        fn, jn = euler_angles_rad.shape
        
        # breakpoint()
        test = self.absolute_quat(euler_angles_rad)
        #new:
        
        return test.reshape(fn, -1, 4)
        
        #test = self.absolute_euler_angles(euler_angles_rad)
        # temp = Rotation.from_euler('ZXY', euler_angles_rad.reshape(-1, 3), degrees=True)
        # temp = temp.as_euler('XYZ', degrees=False)
        # temp = self.quat_from_euler_xyz(temp[:,0], temp[:,1], temp[:,2])
        # temp = temp.reshape(fn, -1, 4)
        
        # return temp
        # #end new (replaced as_quat)
        # # return Rotation.from_euler('ZXY', euler_angles_rad.reshape(-1, 3), degrees=True).as_quat().reshape(fn, -1, 4)
    

    def get_euler_angles(self, quats):
        fn, jn, df = quats.shape
        return Rotation.from_quat(quats.reshape(-1, 4)).as_euler('ZXY').reshape(fn, -1, 3)
    
    def get_euler_angles_from_rots(self, rots):
        fn, jn, mee = rots.shape
        return Rotation.from_matrix(rots.reshape(-1, 9).reshape(-1, 3, 3)).as_euler('ZXY', degrees=True).reshape(fn, -1, 3)
    
    def get_rotation_matrices(self):
        euler_angles_rad = self.get_rotations()
        fn, jn = euler_angles_rad.shape
        return Rotation.from_euler('ZXY', euler_angles_rad.reshape(-1, 3), degrees=True).as_matrix().reshape(-1, 9).reshape(fn, -1, 9)
       
    # Motion data has to be of shape (frames, number of joints * 3 + 3 channels for root positions)
    def compute_world_positions_given(self, motion_data):
        frames = len(motion_data)
        def compute_joint_world_position(joint, frame_data):
            if joint.parent:
                parent_transform = compute_joint_world_position(joint.parent, frame_data)
                rotation = euler_to_rot_matrix(frame_data[joint.motion_index:joint.motion_index+3])
                
            else:
                parent_transform = np.eye(4)
                rotation = euler_to_rot_matrix(frame_data[joint.motion_index+3:joint.motion_index+6])
                joint.offset = frame_data[joint.motion_index:joint.motion_index+3]
            
            joint_transform = np.eye(4)
            joint_transform[:3, :3] = rotation
            joint_translate = np.eye(4)
            joint_translate[:3, 3] = joint.offset
            return parent_transform @ joint_translate @ joint_transform

        world_positions = np.zeros((frames, self.total_channels // 3 - 1, 3))
        for frame_index in range(frames):
            frame_data = motion_data[frame_index]
            for joint_name, joint in self.joints.items():
                #print()
                #print(joint_name)
                #print()
                if joint.motion_index is not None:
                    #joint_pos = compute_joint_world_position(joint, frame_data)
                    joint_pos = compute_joint_world_position(joint, frame_data)[:3, 3]
                    if joint.parent:
                        world_positions[frame_index, joint.motion_index // 3 - 1] = joint_pos
                    else:
                        world_positions[frame_index, joint.motion_index // 3] = joint_pos
        return world_positions
