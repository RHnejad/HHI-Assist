import os
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from model import ModelMain 
from utils.hhi_2 import HHI2


parser = argparse.ArgumentParser(description="Arguments for running the script")
parser.add_argument("--data-dir", type=str, default='/NewData/Cleaned')
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--input_n", type=int, default=24)
parser.add_argument("--output_n", type=int, default=24)
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                    help='Choose to train or test from the model')
parser.add_argument('--output_dir', type=str, default='default')
parser.add_argument('--resume', action='store_true', help='Resume training of a finished training model. If for example you have a model that was trained for 20 epochs, and you want to resume training it for 5 more epochs, you add --resume --epochs 25 to your command line.')
parser.add_argument('--joints', type=int, default=20, help='Number of joints in the dataset. Default is 20 for HHI dataset')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--name', type=str, default='results')
parser.add_argument('--sample_rate', type=float, default=0.2)
parser.add_argument('--layers', type=int, default=4)
parser.add_argument('--lr', type=float, default=1.0e-3)
parser.add_argument('--ov', type=int, default=8, help='Overlap factor. Default value is 8. Overlap factor must divide frame rate')
parser.add_argument('--num_steps', type=int, default=50, help='Number of sampling steps the denoiser performs when generating samples.')
parser.add_argument('--mu', type=int, default=1)
parser.add_argument('--two', action='store_true',
                    help='If the model will take as input pairs of sequences or not')
parser.add_argument('--h', type=int, default=0,
                    help='different configurations when loading two people data. See hhi_2.py for more information')
parser.add_argument('--who', type=str, default=None, choices=[None, "CG", "CR"],
                    help='In 2 people case, who designs the person the model will predict. In 1 person case, refers to which one get loaded in the dataset')
parser.add_argument('--batch', type=int, default=256,
                    help='Batch size')
parser.add_argument('--generalization_experiment', action='store_true',
                    help='TRI Dataset Split becomes : train => Lay to Seated, Sit to Stand  , test => Floor to Standing')

args = parser.parse_args()
new_args = vars(args)

config = {
    'train':
        {
            'epochs': args.epochs,
            'batch_size': args.batch,
            'batch_size_test': args.batch,
            'lr': args.lr
        },
    'diffusion':
        {
            'layers': args.layers,
            'channels': 64,
            'nheads' : 8,
            'diffusion_embedding_dim': 128, 
            'beta_start': 0.0001,
            'beta_end': 0.5,
            'num_steps': args.num_steps,
            'schedule': "cosine",
        },
    'model':
        {
            'is_unconditional': 0,
            'timeemb': 128,
            'featureemb': 16
        }
}


def save_csv_log(head, value, is_create=False, file_name='test'):
    if len(value.shape) < 2:
        value = np.expand_dims(value, axis=0)
    df = pd.DataFrame(value)
    file_path = f'{output_dir}/{file_name}.csv'
    if not os.path.exists(file_path) or is_create:
        df.to_csv(file_path, header=head, index=False)
    else:
        with open(file_path, 'a') as f:
            df.to_csv(f, header=False, index=False)


def save_state(model, optimizer, scheduler, epoch_no, foldername):
    params = {'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch_no}
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), foldername + "/model.pth")
    else:
        torch.save(model.state_dict(), foldername + "/model.pth")

    torch.save(params, foldername + "/params.pth")

def train(
        model,
        config,
        train_loader,
        valid_loader=None,
        valid_epoch_interval=5,
        foldername="",
        load_state=False
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    if load_state:
        optimizer.load_state_dict(torch.load(f'{output_dir}/params.pth')['optimizer'])

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1,p2], gamma=0.1
    )

    if load_state:
        lr_scheduler.load_state_dict(torch.load(f'{output_dir}/params.pth')['scheduler'])

    train_loss = []
    valid_loss = []
    train_loss_epoch = []
    valid_loss_epoch = []
  

    train_dataset = HHI2(0, **vars(args))
    print('>>> Training dataset length: {:d}'.format(train_dataset.__len__()))
    valid_dataset = HHI2(1, **vars(args))
    print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0,
                                pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0,
                        pin_memory=True)

    best_valid_loss = 1e10
    start_epoch = 0
    if load_state:
        start_epoch = torch.load(f'{output_dir}/params.pth')['epoch']
    for epoch_no in range(start_epoch, config["epochs"]):
        
        # We choose our data loader for incremental training depending on the epoch

        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                batch = train_batch
                
                optimizer.zero_grad()

                loss = model(batch).mean()
                loss.backward()
                avg_loss += loss.item()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm= 1.0)
                
                optimizer.step()
                it.set_postfix( 
                    ordererd_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()
        train_loss.append(avg_loss / batch_no)
        train_loss_epoch.append(epoch_no)
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        batch = valid_batch 
                        loss = model(batch, is_train = 0).mean()
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh = False,
                        )

            valid_loss.append(avg_loss_valid / batch_no)
            valid_loss_epoch.append(epoch_no)
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    '\n best loss is updated to ',
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )
                save_state(model, optimizer, lr_scheduler, epoch_no, foldername)

            if (epoch_no + 1) == config["epochs"]:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.plot(train_loss_epoch, train_loss)
                ax.plot(valid_loss_epoch, valid_loss)
                ax.grid(True)
                plt.show()
                fig.savefig(f"{foldername}/loss.png")

    save_state(model, optimizer, lr_scheduler, config["epochs"], foldername)
    np.save(f'{foldername}/train_loss.npy', np.array(train_loss))
    np.save(f'{foldername}/valid_loss.npy', np.array(valid_loss))


def mpjpe_error(batch_imp, batch_gt, skel_info_placeholder=None):
    batch_imp = batch_imp.contiguous().view(-1, 3)
    batch_gt = batch_gt.contiguous().view(-1, 3)
    # When converting numpy arrays to torch tensors, although it is float32, torch will do computations with only 8 digits
    torch.set_printoptions(16)
    return torch.mean(torch.norm(batch_gt - batch_imp, 2, 1))



def _compute_world_positions(pred_euler, orig_euler, parser, permute=None):
    pred_pos = parser.compute_world_positions_given(pred_euler)
    orig_pos = parser.compute_world_positions_given(orig_euler)
    if permute is not None:
        pred_pos = pred_pos[:, -20:, :][:, permute, :]
        orig_pos = orig_pos[:, -20:, :][:, permute, :] #TBV 
    else:
        pred_pos = pred_pos[:, -20:, :]
        orig_pos = orig_pos[:, -20:, :]
    return pred_pos, orig_pos

def evaluate(model, loader, nsample =5, scaler=1, sample_strategy='best'):
    
    torch.set_default_dtype(torch.float32)

    end_nr = 3 
    end_nr *= 2 if args.two else 1

    loss_fn = mpjpe_error
    # top_samples = []
    # bottom_samples = []
    curr_best_mpjpe = float("inf")
    curr_worst_mpjpe = float("-inf")
    loss_total, evalpoints_total = 0, 0

    with torch.no_grad():
        model.eval()

        all_target, all_observed_time, all_evalpoint, all_generated_samples = [], [], [], []
        titles = np.array(range(output_n)) + 1
        m_p3d_hhi, a_p3d_hhi = np.zeros([output_n]), np.zeros([output_n])
        n = 0
        
        with tqdm(loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                batch = test_batch
                batch_size = batch["pose"].shape[0]
                n += batch_size

                if isinstance(model, nn.DataParallel):
                    output = model.module.evaluate(batch, nsample, args.mu)
                else:
                    output = model.evaluate(batch, nsample, args.mu)

                samples, c_target, eval_points, observed_time = output 
                samples, c_target = samples.permute(0, 1, 3, 2), c_target.permute(0, 2, 1)

                key = "pose"
                samples_mean = np.mean(samples.cpu().numpy(), axis=1)
                renorm_pose, renorm_c_target = [], []
                #print("c_target", c_target[0])
                # len(sample_mean) is equivalent to batch size
                for i in range(len(samples_mean)): 
                    renorm_c_target_i = c_target.cpu().data.numpy()[i][input_n:input_n + output_n]
                    if sample_strategy == 'best':
                        best_renorm_pose = None
                        best_error = float('inf')
                        for j in range(nsample):
                            renorm_pose_j = samples.cpu().numpy()[i][j][input_n:]
                            error = loss_fn(torch.from_numpy(renorm_pose_j).view(output_n, args.joints, end_nr),
                                                torch.from_numpy(renorm_c_target_i).view(output_n, args.joints, end_nr), batch[key][i])
                            if error.item() < best_error:
                                best_error = error.item()
                                best_renorm_pose = renorm_pose_j
                    else:
                        best_renorm_pose = samples_mean[i][input_n:input_n + output_n]
                    renorm_pose.append(best_renorm_pose)
                    renorm_c_target.append(renorm_c_target_i)
                renorm_pose = torch.from_numpy(np.array(renorm_pose))
                renorm_c_target = torch.from_numpy(np.array(renorm_c_target))
                eval_points = eval_points[:, input_n:input_n + output_n, :]
                
                # When training a model that takes two sequences as inputs, first sequence is CareGiver.
                # Therefore, if we are predicting CG and we want to evaluate only CG, we take the first 3 xyz coordinates

                if args.two and args.who == "CG":
                    unique = [0,1,2] 
                elif args.two and args.who == "CR":
                    unique = [3,4,5]
                elif args.two:
                    unique = [0,1,2,3,4,5]
                else:
                    unique = [0,1,2]

                torch.set_printoptions(precision=8)
                
                loss_p3d_hhi = torch.sum(torch.mean(torch.norm(renorm_c_target.view(-1, output_n, args.joints, end_nr)[:,:,-20:,unique]
                                                                - renorm_pose.view(-1, output_n, args.joints, end_nr)[:,:,-20:,unique], dim=3),
                                                                dim=2), dim=0) #-20: for excluding the hip joint

                # This is for computing MPJPJE when predicting caregiver and carereceiver simultaneously
                if unique == [0,1,2,3,4,5]:
                    interm1 = renorm_c_target.view(-1, output_n, args.joints, end_nr)[:,:,-20:,unique]
                    interm2 = renorm_pose.view(-1, output_n, args.joints, end_nr)[:,:,-20:,unique]
                    loss_p3d_hhi = torch.sum(torch.mean(torch.mean(torch.norm(interm1.view(-1, output_n, 20, 2, 3)
                                                                - interm2.view(-1, output_n, 20, 2, 3), dim=4), dim=3),
                                                                dim=2), dim=0)

                m_p3d_hhi += loss_p3d_hhi.cpu().data.numpy()
                all_target.append(renorm_c_target)
                all_evalpoint.append(eval_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(renorm_pose)

                loss_current = loss_fn(renorm_pose.view(-1, output_n, args.joints, end_nr)[:,:,-20:,unique],
                                            renorm_c_target.view(-1, output_n, args.joints, end_nr)[:,:,-20:,unique], batch[key][i])

                if (loss_current > curr_worst_mpjpe):
                    curr_worst_mpjpe = loss_current
                    bottom_samples = [samples, c_target]
                if (loss_current < curr_best_mpjpe):
                    curr_best_mpjpe = loss_current
                    top_samples = [samples, c_target]
                loss_total += loss_current.item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "average_mpjpe": loss_total / batch_no,
                        "batch_no": batch_no
                    },
                    refresh=True,
                )
                

            ret = {}
            m_p3d_hhi = m_p3d_hhi / n
            for j in range(output_n):
                ret["#{:d}".format(titles[j])] = m_p3d_hhi[j]

            print("total_mpjpe", np.mean(m_p3d_hhi))
            print("mpjpe_in_time_steps", m_p3d_hhi)

            return all_generated_samples, all_target, all_evalpoint, ret
        
if __name__ == "__main__":
    torch.set_printoptions(24)
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: %s' % device)

    data_dir = args.data_dir
    output_dir = f'{args.output_dir}'
    input_n = args.input_n
    output_n = args.output_n
    config['train']['epochs'] = args.epochs

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dimension = 3
    dimension *= 2 if args.two else 1

    model = ModelMain(config, device, target_dim=(args.joints * dimension))


    if args.resume:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(torch.load(f'{output_dir}/model.pth'))
        else:
            model.load_state_dict(torch.load(f'{output_dir}/model.pth'))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    if args.mode == "train":  
        train(
            model,
            config["train"],
            train_loader=None, 
            valid_epoch_interval=5,
            valid_loader=None,
            foldername=output_dir,
            load_state=args.resume
        )
    
    elif args.mode == "test":
        
        test_dataset = HHI2(2, **vars(args))
        print('>>> Test dataset length: {:d}'.format(test_dataset.__len__()))

        test_loader = DataLoader(test_dataset, batch_size=config["train"]["batch_size_test"], shuffle=False,
                                 num_workers=0, pin_memory=True)
        
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(torch.load(f'{output_dir}/model.pth'))
        else:
            model.load_state_dict(torch.load(f'{output_dir}/model.pth'))

        pose, target, mask, ret = evaluate(
            model,
            test_loader,
            nsample=5,
            scaler=1,
            sample_strategy='NOTbest',
        )  

        ret_log = np.array([])
        head = np.array([])
        for k in range(1, output_n + 1):
            head = np.append(head, [f'#{k}'])
        for k in ret.keys():
            ret_log = np.append(ret_log, [ret[k]])
        save_csv_log(head, ret_log, is_create=True, file_name=f'fde_{args.name}')