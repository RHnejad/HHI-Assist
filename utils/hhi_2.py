import os
import glob 
import numpy as np
from torch.utils.data import Dataset
from utils.misc import get_info
import pandas as pd
import random
import torch

class HHI2(Dataset):
    def __init__(self, split, **kwargs):
        """
        :param path_to_data
        :param input_nbr_frames
        :param output_nbr_frames
        :param task_chosen
        :param type of data : train 0 test 1 validation 2
        """
        data_dir = kwargs['data_dir']
        input_n = kwargs['input_n']
        output_n = kwargs['output_n']
        sample_rate = kwargs['sample_rate']
        ov_factor = kwargs['ov']
        h = kwargs['h']
        self.in_n = input_n
        self.out_n = output_n 
        self.p3d = {}
        self.data_idx = []
        seq_len = self.in_n + self.out_n 
        #self.frame_n = frame_n
        self.who = kwargs['who']
        self.two = kwargs['two']

        # data_dir = os.getcwd() + data_dir
        data_dir = os.path.join(os.getcwd(), data_dir) if not data_dir.startswith('/') else data_dir


        if kwargs['generalization_experiment']:
            subjects = [
                ["AA-RM", "AB-JB", "BC-CC", "BC-HKG", "BC-JH", "SL-JH",
                                        "BC-LPR", "ED-GG", "ED-MT", "JB-BC", "RM-ED", "SR-LPR",
                                        "SL-LPR", "PH-BC", "ED-EC", "GR-JW", "OP-MF", "KT-MA"],
                ["SC-PH"],
                ["Task3"]
            ]
        else:
            subjects = [
                ["AA-RM", "AB-JB", "BC-CC", "BC-HKG", "BC-JH", "SL-JH", "BC-LPR", "ED-GG", "ED-MT",
                                        "JB-BC", "RM-ED", "SR-LPR", "SL-LPR", "PH-BC", "ED-EC"],
                ["SC-PH"],
                ["GR-JW", "OP-MF", "KT-MA"]
            ]


         
        print(subjects)
        if split == 0: # training
            subs = subjects[0]
        elif split == 1: # validation
            subs = subjects[1]
        elif split == 2: # test
            subs = subjects[2]

        key = 0
        for sub_folder in os.listdir(data_dir):
            if sub_folder in subs:
                # print("subfolder", sub_folder)
                data_path = os.path.join(data_dir, sub_folder)
                files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
                files.sort()

                if self.two:
                    for i in range(0, len(files), 2):
                        csv_path = os.path.join(data_path, files[i])
                        csv_path2 = os.path.join(data_path, files[i+1])

                        p1 = get_info(csv_path, sub_folder)
                        p2 = get_info(csv_path2, sub_folder)

                        # ORDER IS CARE RECEIVER FIRST, CARE GIVER SECOND

                        # Read CSV File
                        if p1 == "CR" and p2 == "CG":
                            df = pd.read_csv(csv_path)
                            df2 = pd.read_csv(csv_path2)
                        elif p1 == "CG" and p2 == "CR":
                            df = pd.read_csv(csv_path2)
                            df2 = pd.read_csv(csv_path)
                        elif p1 == "P1" and p2 == "P2":
                            df = pd.read_csv(csv_path)
                            df2 = pd.read_csv(csv_path2)
                        elif p1 == "P2" and p2 == "P1":
                            df = pd.read_csv(csv_path2)
                            df2 = pd.read_csv(csv_path)

                        # print(f"Processing both files: {csv_path} and {csv_path2}")
                        try:
                            df, df2 = self.read_pairs(df, df2, sample_rate, h)
                        except:
                            breakpoint()

                        nr_samples = (int(df.shape[0] / 24) - 2) * ov_factor + 1
                        # print("nr_samples: ", nr_samples)

                        # FINAL ORDER TURNS OUT TO BE CARE GIVER THEN CARE RECEIVER 
                        df2 = df2.to_numpy()
                        w1, w2 = df2.shape 
                        df2 = df2.reshape(w1, int(w2/3), 3)
                        df = df.to_numpy().reshape(w1, int(w2/3), 3)
                        both = np.concatenate((df2, df), 2).reshape(w1, 2*w2)

                        if (nr_samples >= 1):
                            npped = both
                            augment1 = [key] * nr_samples
                            augment2 = [int(self.in_n/ov_factor) * i for i in range(nr_samples)]

                            # npped shape is (nr of frame, nr of joints * 3)
                            self.p3d[key] = npped

                            self.data_idx.extend(zip(augment1, augment2))
                            key += nr_samples

                else:
                    for csv_file in files:
                        csv_path = os.path.join(data_path, csv_file)

                        # Read CSV File
                        df = pd.read_csv(csv_path)
                        try:
                            df = self.read_single(df, sample_rate, h)
                        except:
                            breakpoint()               

                        nr_samples = (int(df.shape[0] / 24) - 2) * ov_factor + 1
                        # print("nr_samples: ", nr_samples)

                        # get_info returns whether the motion sequence corresponds to a care giver or a care receiver
                        p1 = get_info(csv_path, sub_folder)
                        who = kwargs['who']
                        # print(who, p1)

                        # if who is not None, either only caregiver or carereceiver motion sequences will be loaded
                        if who != None:
                            if who != p1:
                                nr_samples = 0
                            else:
                                pass
                        
                        if (nr_samples >= 1):
                            npped = df.to_numpy()
                            augment1 = [key] * nr_samples
                            augment2 = [int(self.in_n/ov_factor) * i for i in range(nr_samples)]

                            self.p3d[key] = npped

                            self.data_idx.extend(zip(augment1, augment2))
                            key += nr_samples


        print("length is ",np.shape(self.data_idx)[0])

    def read_single(self, a, sample_rate, hip=0):
        # 0 : hip translation, no hip joint
        # 1 : no hip translation, with hip joint
        # 2 : hip translation, with hip joint

        # Read files
        df = a

        # Drop time columns
        df = df.drop(df.columns[0], axis=1)  

        # Sample down to 24 fps
        df = df.iloc[::int(1/sample_rate)]

        # Do the hip translation
        if hip == 0 or hip == 2:
            subx = [a for a in range(3,78) if a%3 == 0]
            suby = [a for a in range(3,78) if a%3 == 1]
            subz = [a for a in range(3,78) if a%3 == 2]
            for s in subx:
                df.iloc[:, s] -= df.iloc[:, 0]
            for s in suby:
                df.iloc[:, s] -= df.iloc[:, 1]
            for s in subz:
                df.iloc[:, s] -= df.iloc[:, 2]
        

        # idx_toremove contains the indices of the endsites i.e. not joints
        # 0,1,2 are the hip's xyz coordinates
        # 5*3,5*3+1,5*3+2 are the right toe base endsite's xyz coordinates
        # 10*3,10*3+1,10*3+2 are the left toe base endsite's xyz coordinates
        # 17*3,17*3+1,17*3+2 are the right hand endsite's xyz coordinates
        # 22*3, 22*3+1, 22*3+2 are the left hand endsite's xyz coordinates
        # 25*3. 25*3+1, 25*3+2 are the head's endsite's xyz coordinates

        if hip == 0:
            # If we do not want to feed the model the global hip position information, we drop the the 0'th joints XYZ coordinates
            idx_toremove = [0,1,2,5*3,5*3+1,5*3+2,10*3,10*3+1,10*3+2,17*3,17*3+1,17*3+2,22*3,22*3+1,22*3+2,25*3,25*3+1,25*3+2]
        else:
            idx_toremove = [5*3,5*3+1,5*3+2,10*3,10*3+1,10*3+2,17*3,17*3+1,17*3+2,22*3,22*3+1,22*3+2,25*3,25*3+1,25*3+2]

        df = df.drop(df.columns[idx_toremove], axis=1)  

        return df
    
    def read_pairs(self, a, b, sample_rate, hip=0):
        # 0 : hip translation, no hip joint
        # 1 : no hip translation, with hip joint
        # 2 : hip translation, with hip joint
        # 3 : hip translation, with difference of hip joints

        # Read files
        df = a
        df2 = b

        # Drop time columns

        df = df.drop(df.columns[0], axis=1)  
        df2 = df2.drop(df2.columns[0], axis=1)  

        # Sample down to 24 fps

        df = df.iloc[::int(1/sample_rate)]
        df2 = df2.iloc[::int(1/sample_rate)]

        # Do the hip translation

        if hip == 0 or hip == 2 or hip == 3:
            subx = [a for a in range(3,78) if a%3 == 0]
            suby = [a for a in range(3,78) if a%3 == 1]
            subz = [a for a in range(3,78) if a%3 == 2]
            for s in subx:
                df.iloc[:, s] -= df.iloc[:, 0]
                df2.iloc[:, s] -= df2.iloc[:, 0]
            for s in suby:
                df.iloc[:, s] -= df.iloc[:, 1]
                df2.iloc[:, s] -= df2.iloc[:, 1]
            for s in subz:
                df.iloc[:, s] -= df.iloc[:, 2]
                df2.iloc[:, s] -= df2.iloc[:, 2]
            
        # If we want the distance between caregiver and carereceiver given as input to the model instead of global position
        if hip == 3 :
            sub = df.iloc[:,0:3].values - df2.iloc[:,0:3].values
            df.iloc[:,0:3] = sub
            df2.iloc[:,0:3] = sub

        # idx_toremove contains the indices of the endsites i.e. not joints
        # 0,1,2 are the hip's xyz coordinates
        # 5*3,5*3+1,5*3+2 are the right toe base endsite's xyz coordinates
        # 10*3,10*3+1,10*3+2 are the left toe base endsite's xyz coordinates
        # 17*3,17*3+1,17*3+2 are the right hand endsite's xyz coordinates
        # 22*3, 22*3+1, 22*3+2 are the left hand endsite's xyz coordinates
        # 25*3. 25*3+1, 25*3+2 are the head's endsite's xyz coordinates

        if hip == 0:
            # If we do not want to feed the model the global hip position information, we drop the the 0'th joints XYZ coordinates
            idx_toremove = [0,1,2,5*3,5*3+1,5*3+2,10*3,10*3+1,10*3+2,17*3,17*3+1,17*3+2,22*3,22*3+1,22*3+2,25*3,25*3+1,25*3+2]
        else:
            idx_toremove = [5*3,5*3+1,5*3+2,10*3,10*3+1,10*3+2,17*3,17*3+1,17*3+2,22*3,22*3+1,22*3+2,25*3,25*3+1,25*3+2]

        df = df.drop(df.columns[idx_toremove], axis=1)  
        df2 = df2.drop(df2.columns[idx_toremove], axis=1) 

        return df, df2


    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        
        pose = self.p3d[key][fs]

        mask = np.zeros((pose.shape[0], pose.shape[1]))
        mask[0:self.in_n, :] = 1

        mask[self.in_n:self.in_n + self.out_n, :] = 0

        if self.who == "CG":
            # if we predict only caregiver, we remove carereceiver i.e set carereciver mask to 1
            mask = mask.reshape((pose.shape[0], int(pose.shape[1]/6), 6))
            mask[:,:,-3:]=1
            mask = mask.reshape((pose.shape[0], pose.shape[1]))
        elif self.who == "CR":
            mask = mask.reshape((pose.shape[0], int(pose.shape[1]/6), 6))
            mask[:,:,:3]=1
            mask = mask.reshape((pose.shape[0], pose.shape[1]))

        data = {
            "pose": pose,
            "mask": mask,
            "timepoints": np.arange(self.in_n + self.out_n)
        }

        return data