import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import os
from PIL import Image
import io

import matplotlib
from matplotlib.animation import FuncAnimation
    
class Visualizer:
    def __init__(self, len):
        self.COUNT = 0
        self.STAT = np.zeros(19)
        self.list_ = np.zeros((len,19,24))
        
        self.skeleton = [[8,0],[0,1],[1,2],[2,3],[8,4],[4,5],[5,6],[6,7],[8,9],[18,10],[10,11],[11,12],
                            [12,13],[18,14],[14,15],[15,16],[16,17],[9,18],[18,19]]
        self.color_pairs = [
            ((100/255, 140/255, 140/255), (40/255, 80/255, 80/255)),   # Dark Teal
            ((130/255, 180/255, 210/255), (70/255, 120/255, 150/255)),   # Blueish Tone
            ((210/255, 160/255, 180/255), (150/255, 100/255, 120/255))  # Pink 
        ]
        self.max_all_xyz = None
        self.min_all_xyz = None

    def _setup_axes(self, ax):
        ax.view_init(elev=80, azim=-90)
        ax.set_box_aspect([(self.max_all_xyz[0]-self.min_all_xyz[0])/(self.max_all_xyz[1]-self.min_all_xyz[1]),
                            1,
                            (self.max_all_xyz[2]-self.min_all_xyz[2])/(self.max_all_xyz[1]-self.min_all_xyz[1])])
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(False)
        ax.set_axis_off()
        ax.axes.set_xlim3d(left=self.min_all_xyz[0], right=self.max_all_xyz[0]) 
        ax.axes.set_ylim3d(bottom=self.min_all_xyz[1], top=self.max_all_xyz[1]) 
        ax.axes.set_zlim3d(bottom=self.min_all_xyz[2], top=self.max_all_xyz[2]) 
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.margins(0)

    def _plot_skeleton(self, ax, data, label, title, seq_number=1):
        xdata, ydata, zdata = data.T
        ax.scatter(xdata, ydata, zdata, color="black", s=0.5)
        n_joints = xdata.shape[0]
        l_j = 10 if seq_number == 1 else 13
        for j in range(n_joints-1):
            color_indx = seq_number % len(self.color_pairs) - 1
            if seq_number > 2:
                color_indx = 2
            selected_pair = self.color_pairs[color_indx]
            color = selected_pair[0] if j < l_j else selected_pair[1]
            ax.plot(xdata[self.skeleton[j]], ydata[self.skeleton[j]], zdata[self.skeleton[j]], color=color)
        self._setup_axes(ax)

    def visualize_sequence(self, sequences, save_path, string="test_vis_seq", return_array=False, project_under=False, title=None):
        if not sequences:
            raise ValueError("No sequences provided")
        n_columns = 6
        total_frames = sum(seq.shape[0] for seq in sequences)
        skip = max([seq.shape[0] for seq in sequences]) // n_columns
        sequences_subsampled = [seq[::skip,:,:] for seq in sequences]
        sequences_subsampled = [seq[:,1:,:]-seq[:,:1,:] for seq in sequences_subsampled]
        sequences_subsampled = [np.concatenate((np.zeros((seq.shape[0],1,3)),seq), axis=1) for seq in sequences_subsampled]
        n_rows = sum((seq.shape[0] - 1) // n_columns + 1 for seq in sequences_subsampled)
        fig = plt.figure(figsize=(n_columns, n_rows))
        all_xdata = np.concatenate([seq[:, :, 0].flatten() for seq in sequences_subsampled])
        all_ydata = np.concatenate([seq[:, :, 1].flatten() for seq in sequences_subsampled])
        all_zdata = np.concatenate([seq[:, :, 2].flatten() for seq in sequences_subsampled])
        self.max_all_xyz = [np.max(all_xdata), np.max(all_ydata), np.max(all_zdata)]
        self.min_all_xyz = [np.min(all_xdata), np.min(all_ydata), np.min(all_zdata)]
        current_frame = 0
        for seq_idx, seq in enumerate(sequences_subsampled):
            for frame_idx in range(seq.shape[0]):
                ax = fig.add_subplot(n_rows, n_columns, current_frame + 1, projection='3d')
                if project_under:
                    under_seq = sequences_subsampled[1]
                    self._plot_skeleton(ax, under_seq[frame_idx, :, :], label=f"pose|{current_frame}", title=f"frame {current_frame}", seq_number=1)
                self._plot_skeleton(ax, seq[frame_idx, :, :], label=f"pose|{current_frame}", title=f"frame {current_frame}", seq_number=seq_idx + 1)
                current_frame += 1
            while current_frame % n_columns != 0:
                fig.add_subplot(n_rows, n_columns, current_frame + 1)
                plt.axis('off')
                current_frame += 1
        plt.subplots_adjust(wspace=0, hspace=0)
        if return_array:
            fig_aux = plt.figure(figsize=(1, n_rows))
            for i in range(len(sequences)):
                ax = fig_aux.add_subplot(n_rows, 1, i + 1, projection='3d')
                for k in range(sequences[i].shape[0]):
                    self._plot_skeleton(ax, sequences[i][k, 0], label="", title="", seq_number=i + 1)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            image = Image.open(buf)
            image_array = np.array(image)
            buf.close()
            buf_aux = io.BytesIO()
            fig_aux.savefig(buf_aux, format='png', bbox_inches='tight', pad_inches=0)
            buf_aux.seek(0)
            image_aux = Image.open(buf_aux)
            image_aux_array = np.array(image_aux)
            buf_aux.close()
            image_array = np.concatenate((image_array, image_aux_array), axis=1)
            plt.close()
            return image_array
        else:
            parts = string.split('/')
            directory = os.path.join(save_path,"seq", *parts[:-1])
            os.makedirs(directory, exist_ok=True)
            if title is not None:
                assert type(title) == str, "Title must be a string"
                plt.title(title)
            plt.savefig(os.path.join(directory, "{}.png".format(parts[-1])), bbox_inches='tight', pad_inches=0)
            plt.savefig(os.path.join(directory, "{}.pdf".format(parts[-1])), bbox_inches='tight', pad_inches=0)
            plt.close()

    def vis_link_len(self, sequences, save_path, string="test_link_len"):
        def get_or_create_subplot(fig, nrows, ncols, index):
            row = (index - 1) // ncols
            col = (index - 1) % ncols
            for ax0 in fig.get_axes():
                if ax0.get_subplotspec().rowspan.start == row and ax0.get_subplotspec().colspan.start == col:
                    return ax0
            return fig.add_subplot(nrows, ncols, index)
        
        fig = plt.figure(figsize=(20, 15))
        times = np.linspace(-23, 24, num=48, dtype=int)
        n_s = sequences.shape[0] if len(sequences.shape) == 4 else 1
        
        # Calculate global min and max for y-axis limits
        global_min = float('inf')
        global_max = float('-inf')
        
        for j in range(n_s):
            self.COUNT += 1
            sequence = sequences[j] if n_s > 1 else sequences
            for i in range(len(self.skeleton)):
                joint_1 = sequence[:, self.skeleton[i][0], :]
                joint_2 = sequence[:, self.skeleton[i][1], :]
                link_len = np.linalg.norm(joint_1 - joint_2, axis=-1)
                l = link_len[0]
                link_len = link_len - l
                global_min = min(global_min, np.min(link_len * 1000))
                global_max = max(global_max, np.max(link_len * 1000))
        
        for j in range(n_s):
            sequence = sequences[j] if n_s > 1 else sequences
            for i in range(len(self.skeleton)):
                joint_1 = sequence[:, self.skeleton[i][0], :]
                joint_2 = sequence[:, self.skeleton[i][1], :]
                link_len = np.linalg.norm(joint_1 - joint_2, axis=-1)
                ax = get_or_create_subplot(fig, 4, 5, i + 1)
                l = link_len[0]
                link_len = link_len - l
                
                self.STAT[i] += np.sum(np.abs(link_len[24:])) * 1000 / 24
                ax.plot(times, link_len * 1000, linewidth=2, color="steelblue", alpha=0.2)
                ax.set_title(f"link {i}")
                ax.set_xlabel('time step', fontsize=18)
                ax.set_ylabel('link length [mm]', fontsize=16)
                ax.tick_params(axis='both', which='major', labelsize=16)
                ax.tick_params(axis='both', which='minor', labelsize=14)
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
                x_ticks = np.linspace(-23, 24, num=5, dtype=int)
                ax.set_xticks(x_ticks)
                ax.set_ylim(global_min, global_max)  # Set the same y-axis limits for all subplots
        
        plt.tight_layout()
        parts = string.split('/')
        directory = os.path.join(save_path, "links_new", *parts[:-1])
        os.makedirs(directory, exist_ok=True)
        plt.savefig(os.path.join(directory, "{}_.png".format(parts[-1])), pad_inches=0)
        plt.savefig(os.path.join(directory, "{}_.pdf".format(parts[-1])), pad_inches=0)
        plt.close()
        
        if self.COUNT > 10510 or self.COUNT % 1000 == 0:
            print(f"Average link length change per joint: {self.STAT / self.COUNT} mm")
            print(f"Average link length change: {(np.mean(self.STAT) / self.COUNT):.6f} mm")

    # def vis_link_len_old(self, sequences, save_path, string="test_link_len"):
    #     def get_or_create_subplot(fig, nrows, ncols, index):
    #         row = (index - 1) // ncols
    #         col = (index - 1) % ncols
    #         for ax0 in fig.get_axes():
    #             if ax0.get_subplotspec().rowspan.start == row and ax0.get_subplotspec().colspan.start == col:
    #                 return ax0
    #         return fig.add_subplot(nrows, ncols, index)
    #     fig = plt.figure(figsize=(20, 15))
    #     times = np.linspace(-23, 24, num=48, dtype=int)
    #     n_s = sequences.shape[0] if len(sequences.shape) == 4 else 1
    #     for j in range(n_s):
    #         self.COUNT += 1
    #         sequence = sequences[j] if n_s > 1 else sequences
    #         for i in range(len(self.skeleton)):
    #             joint_1 = sequence[:,self.skeleton[i][0],:]
    #             joint_2 = sequence[:,self.skeleton[i][1],:]
    #             link_len = np.linalg.norm(joint_1-joint_2, axis=-1)
    #             ax = get_or_create_subplot(fig, 4, 5, i + 1)
    #             l = link_len[0]
    #             link_len = link_len - l
                
    #             # self.list_[self.COUNT-1, i] = link_len[24:]*1000
                
    #             self.STAT[i] += np.sum(np.abs(link_len[24:]))*1000 / 24
    #             ax.plot(times, link_len*1000, linewidth=2, color = "steelblue", alpha=0.2)
    #             ax.set_title(f"link {i}")
    #             ax.set_xlabel('time step', fontsize=18)
    #             ax.set_ylabel('link length [mm]', fontsize=16)
    #             ax.tick_params(axis='both', which='major', labelsize=16)
    #             ax.tick_params(axis='both', which='minor', labelsize=14)
    #             ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    #             ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    #             x_ticks = np.linspace(-23, 24, num=5, dtype=int)
    #             ax.set_xticks(x_ticks)
    #     plt.tight_layout()
    #     parts = string.split('/')
    #     directory = os.path.join(save_path, "links_new", *parts[:-1])
    #     os.makedirs(directory, exist_ok=True)
    #     plt.savefig(os.path.join(directory, "{}_.png".format(parts[-1])), pad_inches=0)
    #     plt.savefig(os.path.join(directory, "{}_.pdf".format(parts[-1])), pad_inches=0)
    #     plt.close()
    #     if self.COUNT > 10510 or self.COUNT % 1000 == 0:
    #         print(f"Average link length change per joint: {self.STAT/self.COUNT} mm")
    #         print(f"Average link length change: { (np.mean(self.STAT)/self.COUNT):.6f} mm")

    def cal_abs_diff(self, x, y_preds):
        diff_sum = np.empty((len(self.skeleton), y_preds.shape[1]))
        for i in range(len(self.skeleton)):
            joint_1 = x[:,self.skeleton[i][0],:]
            joint_2 = x[:,self.skeleton[i][1],:]
            link_len_ref = np.linalg.norm(joint_1-joint_2, axis=-1)
            link_len_ref = np.expand_dims(link_len_ref, axis=1)
            joint_1 = y_preds[:,:,self.skeleton[i][0],:]
            joint_2 = y_preds[:,:,self.skeleton[i][1],:]
            link_len = np.linalg.norm(joint_1-joint_2, axis=-1)
            diff = np.abs(link_len_ref-link_len)
            diff_sum[i] = np.sum(diff, axis=0)
        return diff_sum
    
    def report_and_plot_stats(self, save_path):
        # Create the links_new directory if it doesn't exist
        links_dir = os.path.join(save_path, "links_new")
        os.makedirs(links_dir, exist_ok=True)

        self.list_abs = np.abs(self.list_)
        
        print(f"___ Average link length change per joint: {self.STAT/self.COUNT} mm")
        print(f"___ Average link length change: { (np.mean(self.STAT)/self.COUNT):.6f} mm")
        
        #save the stats and count arrays
        np.save(os.path.join(links_dir, "stats.npy"), self.STAT)
        np.save(os.path.join(links_dir, "count.npy"), self.COUNT)
        
        
        # #average and variance of self.list_:
        avg = np.mean(self.list_)#, axis=0)
        var = np.var(self.list_)#, axis=0)
        print(f"Average link length change per joint: {avg} mm")
        print(f"Variance of link length change per joint: {var} mm^2")
        
        #average over all:
        avg_all = np.mean(self.list_abs)
        var_all = np.var(self.list_abs)
        print(f"Average link length change: {avg_all:.6f} mm")
        print(f"Variance of link length change: {var_all:.6f} mm^2")
        
        #save the list and list_abs:
        np.save(os.path.join(links_dir, "list.npy"), self.list_)
        np.save(os.path.join(links_dir, "list_abs.npy"), self.list_abs)
        
        
        # breakpoint()
        
        # plot a histogram of the link length change:
        fig = plt.figure(figsize=(20, 15))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(self.list_.flatten(), bins=100, color = "steelblue", alpha=0.7)
        ax.set_title("Histogram of link length change", fontsize=20)
        ax.set_xlabel('link length change [mm]', fontsize=18)
        ax.set_ylabel('frequency', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "links_new", "histogram.png"), pad_inches=0)
        plt.savefig(os.path.join(save_path, "links_new", "histogram.pdf"), pad_inches=0)
        plt.close()
        
        # plot a histogram of ABS the link length change:
        fig = plt.figure(figsize=(20, 15))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(self.list_abs.flatten(), bins=50, color = "steelblue", alpha=0.7)
        ax.set_title("Histogram of link length change", fontsize=20)
        ax.set_xlabel('link length change [mm]', fontsize=18)
        ax.set_ylabel('frequency', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "links_new", "histogram_abs.png"), pad_inches=0)
        plt.savefig(os.path.join(save_path, "links_new", "histogram_abs.pdf"), pad_inches=0)
        plt.close()      
        
        #plot histogram for each joint:
        fig = plt.figure(figsize=(20, 15))
        for i in range(19):
            ax = fig.add_subplot(4, 5, i + 1)
            ax.hist(self.list_[:,i,:].flatten(), bins=100, color = "steelblue", alpha=0.7)
            ax.set_title(f"link {i}")
            ax.set_xlabel('link length change [mm]', fontsize=18)
            ax.set_ylabel('frequency', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=14)
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "links_new", "a_histogram_per_link.png"), pad_inches=0)
        plt.savefig(os.path.join(save_path, "links_new", "a_histogram_per_link.pdf"), pad_inches=0)
        plt.savefig(os.path.join(save_path, "links_new", "a_histogram_per_link.svg"), pad_inches=0)
        plt.close()
            
        #plot subplots for each joint and save the plot:
        fig = plt.figure(figsize=(20, 15))
        times = np.linspace(0, 24, num=24, dtype=int)
        
        avg_per_link = np.mean(self.list_abs, axis=0)
        var_per_link = np.var(self.list_abs, axis=0)
                
        for i in range(19):
            ax = fig.add_subplot(4, 5, i + 1)
            ax.plot(times, avg_per_link[i].T, linewidth=2, color = "steelblue", alpha=0.2)
            ax.set_title(f"link {i}")
            ax.set_xlabel('time step', fontsize=18)
            ax.set_ylabel('link length [mm]', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=14)
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
            x_ticks = np.linspace(0, 24, num=5, dtype=int)
            ax.set_xticks(x_ticks)
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "links_new", "a_all_links.png"), pad_inches=0)
        plt.savefig(os.path.join(save_path, "links_new", "a_all_links.pdf"), pad_inches=0)
        
        plt.close()
        
    def visualize_sequence_gif(self, sequences, save_path, string):
        matplotlib.use("Agg")
        if not sequences:
            raise ValueError("No sequences provided")
        sequences_subsampled = [seq[:,1:,:]-seq[:,:1,:] for seq in sequences]
        sequences_subsampled = [np.concatenate((np.zeros((seq.shape[0],1,3)),seq), axis=1) for seq in sequences_subsampled]
        max_frames = max(seq.shape[0] for seq in sequences_subsampled)
        n_r = len(sequences_subsampled)
        fig, axes = plt.subplots(1, len(sequences_subsampled), subplot_kw={'projection': '3d'}, figsize=(2*n_r, 4))
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
        all_xdata = np.concatenate([seq[:, :, 0].flatten() for seq in sequences_subsampled])
        all_ydata = np.concatenate([seq[:, :, 1].flatten() for seq in sequences_subsampled])
        all_zdata = np.concatenate([seq[:, :, 2].flatten() for seq in sequences_subsampled])
        self.max_all_xyz = [np.max(all_xdata), np.max(all_ydata), np.max(all_zdata)]
        self.min_all_xyz = [np.min(all_xdata), np.min(all_ydata), np.min(all_zdata)]
        
        # Make axes iterable even if there's only one sequence
        if len(sequences_subsampled) == 1:
            axes = [axes]
        
        def update(frame):
            for i, (ax, seq_) in enumerate(zip(axes, sequences_subsampled)):
                ax.clear()
                if frame < seq_.shape[0]:
                    # Check if this is a combined skeleton (two people)
                    n_joints = seq_.shape[1]
                    if n_joints > 20:  # If we have more than 20 joints, it's likely two people combined
                        # Plot each person separately
                        n_joints_per_person = n_joints // 2
                        # First person
                        self._plot_skeleton(ax, seq_[frame, :n_joints_per_person, :], 
                                          label=f"person1|{frame}", title=f"frame {frame}", seq_number=1)
                        # Second person
                        self._plot_skeleton(ax, seq_[frame, n_joints_per_person:, :], 
                                          label=f"person2|{frame}", title=f"frame {frame}", seq_number=2)
                    else:
                        # Normal case - single person
                        self._plot_skeleton(ax, seq_[frame, :, :], label=f"pose|{frame}", title=f"frame {frame}", seq_number=i+1)
                    
                    # Set the title for the frame
                    ax.set_title(f"Frame {frame}")
        
        # Create animation with at least one frame to prevent IndexError
        frames = max(1, max_frames)
        ani = FuncAnimation(fig, update, frames=frames, interval=100)
        gif_path = os.path.join(save_path, "gifs", f"{string}.gif")
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        ani.save(gif_path, writer='imagemagick')
        plt.close(fig)