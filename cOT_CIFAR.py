import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.datasets as td
from torchvision.utils import make_grid
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from utils.unet_oai import UNetModel
import utils.fid_score as fs
from utils.utils_FID import *
import ot
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float

torch.manual_seed(4048)
channels = 3
img_size = 32

def get_UNET():
    # use unet
    return UNetModel(
        in_channels=channels,
        out_channels=channels,
        num_res_blocks=2,
        num_classes=10,
        image_size=img_size,
        model_channels=256,
        channel_mult=(1, 2, 2, 2),
        num_heads=4,
        num_head_channels=64,
        attention_resolutions=(16,)
    ).to(device)



def main(args):
  

   

    # Load target samples
    cifar = td.CIFAR10('cifar10', transform=transforms.ToTensor(), download=True)
    M = 50000
    N = M
    data = DataLoader(dataset=cifar, batch_size=M, shuffle=False)
    data = next(iter(data))
    dim = 3072
    data_tmp = data[0].view(M, dim).to(device)
    ground_truth = data_tmp.clone()
    label2 = data[1].to(device)
    label = torch.nn.functional.one_hot(data[1].to(device))

    observation = args.beta * label.clone()
    M2 = 10000

    fid_list1 = []

    net = get_UNET()
    optim = torch.optim.Adam(net.parameters(), lr=2e-4)
    averaged_model = torch.optim.swa_utils.AveragedModel(net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.decay))

    test_z = torch.randn((M2, 3, 32, 32), device=device)
    mse = torch.nn.MSELoss(reduction="sum")
    ntrain = args.n_epochs
    batch = args.batch_data
    batchOT = args.batchOT
    batch_net = args.batch_net
    progress_bar = tqdm(range(ntrain), total=ntrain, position=0, leave=True)
    for k in progress_bar:
        ind = torch.randperm(M, device=device)
        ind = ind[:batch]
        
        for j in range(batch // batchOT):
            ind_tmp = ind[j * batchOT:(j + 1) * batchOT]
            label_tm = label2[ind_tmp].clone()
            gt = ground_truth[ind_tmp].clone()
            z = torch.randn((batchOT, dim), device=device)
        
            target = torch.cat((gt, observation[ind_tmp]), 1)
            source = torch.cat((z, observation[ind_tmp]), 1)
            t = torch.rand((batchOT,), device=device)
            
            u, v = ot.unif(source.shape[0]), ot.unif(target.shape[0])
            C = torch.cdist(source, target) ** 2
            C = C.cpu().numpy()
            plan = ot.emd(u, v, C)
            ind2 = torch.tensor(np.argmax(plan, 1), device=device)
            g_x = z - gt[ind2]

        
            start = z - t.view(-1, 1) * g_x
            
            start = start.detach()
            start = start.reshape(batchOT, channels, img_size, img_size)
            for p in range(batchOT // batch_net):
                optim.zero_grad()
                loss = mse(net(start[p * batch_net:(p + 1) * batch_net], t[p * batch_net:(p + 1) * batch_net],
                                label_tm[p * batch_net:(p + 1) * batch_net]).reshape(batch_net, dim), g_x[p * batch_net:(p + 1) * batch_net])
                l = loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optim.step()
                if k > 0:
                    averaged_model.update_parameters(net)
                descr = f"Loss={l:.4f}."
                progress_bar.set_description(descr)
    
        if k % 20 == 0 and k != 0:
            averaged_model.eval()
            fid = calc_FID(averaged_model, M2, test_z.clone())
            print("1", fid)
            fid_list1.append((k, fid))
            averaged_model.train()
        
            torch.save(averaged_model.state_dict(), f"{args.dir_name}/nets{args.beta}/net_{k}.pt")
        
    print(fid_list1)
    torch.save(averaged_model.state_dict(), f"{args.dir_name}/nets{args.beta}/net_{k}.pt")
    torch.save(net.state_dict(), f"{args.dir_name}/nets{args.beta}/net{args.beta}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for the script')
    parser.add_argument('--batch_data', type=int, default=50000, help='Dataloader Batch Size')
    parser.add_argument('--n_epochs', type=int, default=501, help='Number of epochs')
    parser.add_argument('--beta', type=int, default=1, help='Value for beta')
    parser.add_argument('--batchOT', type=int, default=500, help='Batch size for OT computation')
    parser.add_argument('--decay', type=float, default=0.9999, help='Decay value for SWA')
    parser.add_argument('--batch_net', type=int, default=100, help='Batch size for networks')
    parser.add_argument('--dir_name', type=str, default="cOT", help='Directory name for saving files')
    args = parser.parse_args()
    save_dir = os.path.join(args.dir_name, f'nets{args.beta}')
    os.makedirs(save_dir, exist_ok=True)
    main(args)
