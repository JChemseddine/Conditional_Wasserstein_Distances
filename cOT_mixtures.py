import argparse
import numpy as np
import torch
from torch import nn
from utils.Util_mixture import *
from geomloss import SamplesLoss
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import ot

device = torch.device("cuda")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and evaluate the model")
    parser.add_argument("--ntrain", type=int, default=4000, help="Number of training steps")
    parser.add_argument("--ntraj", type=int, default=200, help="Batch size of Sinkhorn")
    parser.add_argument("--beta", type=float, default=100., help="Beta value")
    parser.add_argument("--DIMENSION", type=int, default=5, help="Dimension value")
    parser.add_argument("--directory", type=str, default="imgs", help="Directory for saving images")
    parser.add_argument("--seed", type=int, default=42, help="Seed value")
    parser.add_argument("--steps", type=int, default=10, help="euler steps")
    parser.add_argument("--samples_test", type=int, default=1000, help="samples x for one measurement")
    parser.add_argument("--measurement_test", type=int, default=100, help="number measurements")
    parser.add_argument("--train_samples", type=int, default=10000, help="train samples")

    parser.add_argument("--val_samples", type=int, default=2000, help="validation samples")


    return parser.parse_args()

# Define generator network
class Generator(torch.nn.Module):
    def __init__(self, DIMENSION):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(2*DIMENSION+1, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, DIMENSION)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        output = self.relu(output)
        output = self.fc4(output)
        return output

def make_image(true_samples, pred_samples, img_name, directory='imgs', inds=None):
    cmap = plt.cm.tab20
    range_param = 1.3
    no_params = true_samples.shape[1]
    inds = range(no_params)

    fig, axes = plt.subplots(figsize=[12, 12], nrows=no_params, ncols=no_params, gridspec_kw={'wspace':0., 'hspace':0.})

    for j, ij in enumerate(inds):
        for k, ik in enumerate(inds):
            axes[j, k].get_xaxis().set_ticks([])
            axes[j, k].get_yaxis().set_ticks([])
            if j == k:
                axes[j, k].hist(pred_samples[:, ij], bins=50, color=cmap(0), alpha=0.3, range=(-range_param, range_param))
                axes[j, k].hist(pred_samples[:, ij], bins=50, color=cmap(0), histtype="step", range=(-range_param, range_param))

                axes[j, k].hist(true_samples[:, ij], bins=50, color=cmap(2), alpha=0.3, range=(-range_param, range_param))
                axes[j, k].hist(true_samples[:, ij], bins=50, color=cmap(2), histtype="step", range=(-range_param, range_param))
            else:
                val, x, y = np.histogram2d(pred_samples[:, ij], pred_samples[:, ik], bins=25, range=[[-range_param, range_param], [-range_param, range_param]])
                axes[j, k].contour(val, 8, extent=[x[0], x[-1], y[0], y[-1]], alpha=0.5, colors=[cmap(0)])

                val, x, y = np.histogram2d(true_samples[:, ij], true_samples[:, ik], bins=25, range=[[-range_param, range_param], [-range_param, range_param]])
                axes[j, k].contour(val, 8, extent=[x[0], x[-1], y[0], y[-1]], alpha=0.5, colors=[cmap(2)])

    plt.savefig('./'+directory+'/'+img_name, bbox_inches='tight', pad_inches=0.05)
    plt.close()

def train_and_eval(args):
    ntrain = args.ntrain
    ntraj = args.ntraj
    beta = args.beta
    DIMENSION = args.DIMENSION
    directory = args.directory
    seed = args.seed
    NO_samples = args.samples_test
    NO_y = args.measurement_test
    train_samples = args.train_samples
    steps = args.steps

    val_samples = args.val_samples
    torch.manual_seed(seed)
    np.random.seed(seed)
    forward_map = create_forward_model(scale=0.1, dimension=DIMENSION)
    n_mixtures = 10
    mixture_params = []
    b = 0.1
    for _ in range(n_mixtures):
        mixture_params.append((1./n_mixtures, torch.tensor(np.random.uniform(size=DIMENSION)*2-1, device=device, dtype=torch.float), torch.tensor(0.01, device=device, dtype=torch.float)))
    gen = Generator(DIMENSION).to(device)
    x_train = draw_mixture_dist(mixture_params, train_samples).to(device)
    y_train = forward_pass(x_train, forward_map).to(device)+ b*torch.randn_like(x_train, device = device)
    train_set = torch.cat((x_train,y_train),1)
    dataloader = DataLoader(train_set, batch_size=ntraj,
                        shuffle=True, num_workers=0)
    x_test = draw_mixture_dist(mixture_params, NO_y).to(device)
    y_test = forward_pass(x_test, forward_map).to(device)+ b*torch.randn_like(x_test, device = device)
    
    x_val = draw_mixture_dist(mixture_params, val_samples).to(device)
    y_val = forward_pass(x_val, forward_map).to(device)+ b*torch.randn_like(x_val, device = device)
    optimizer = torch.optim.Adam(gen.parameters(), lr=1e-4)

    loss = nn.MSELoss(reduction= "sum")
    print("number of parameters of generator")
    print(sum(p.numel() for p in gen.parameters()))
    min_loss = 1000000.

    progress_bar = tqdm(range(ntrain), total=ntrain, position=0, leave=True)
    for i in progress_bar:
        for k,x_full in enumerate(dataloader):
            x = x_full[:,:DIMENSION].to(device)
            y = x_full[:,DIMENSION:].to(device)
            t = torch.rand((ntraj,), device=device)
            z = torch.randn((ntraj, DIMENSION), device=device)
            rescaled_zy = torch.cat((z, y*beta), dim=1)
            xy = torch.cat((x, y*beta), dim=1)
            zy = torch.cat((z, y), dim=1)
            u, v = ot.unif(rescaled_zy.shape[0]), ot.unif(xy.shape[0])
            C = torch.cdist(rescaled_zy, xy)**2
            C = C.cpu().numpy()
            plan = ot.emd(u, v, C)
            ind2 = torch.tensor(np.argmax(plan, 1), device=device)
            g_z = zy - xy[ind2]
            zy[:, :DIMENSION] -= t.view(-1, 1) * g_z[:, :DIMENSION]
            net_inp = torch.cat((zy, t.unsqueeze(1).reshape(ntraj, 1)), dim=1)
            outp = gen(net_inp)
            optimizer.zero_grad()
            l = loss(outp, g_z[:, :DIMENSION])
            l.backward()
            optimizer.step()

        with torch.no_grad():
            t = torch.rand((val_samples,),device=device)
            # latent distribution
            z = torch.randn((val_samples,DIMENSION),device=device)
            rescaled_zy = torch.cat((z,y_val*beta),dim=1)
            xy = torch.cat((x_val,y_val*beta),dim=1)
            zy = torch.cat((z,y_val),dim=1)

            u,v= ot.unif(zy.shape[0]), ot.unif(xy.shape[0])
            C = torch.cdist(rescaled_zy,xy)**2
            C = C.cpu().numpy()
            plan = ot.emd(u,v,C)
            ind2 = torch.tensor(np.argmax(plan,1),device=device)
            g_z = rescaled_zy -xy[ind2]

            zy[:,:DIMENSION]-= t.view(-1,1)*g_z[:,:DIMENSION]
            net_inp = torch.cat((zy,t.unsqueeze(1).reshape(val_samples,1)),dim=1)   
            outp = gen(net_inp)
            l = loss(outp, g_z[:,:DIMENSION])
            if l < min_loss:
                min_loss = l
                torch.save(gen.state_dict(), 'ot_flow_mixtures.pt')
        descr = f"Loss={min_loss:.4f}."

        progress_bar.set_description(descr)


    Wloss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    torch.manual_seed(seed)
    gen = Generator(DIMENSION).to(device)

    gen.load_state_dict(torch.load('ot_flow_mixtures.pt'))

    w2_mean = 0
    with torch.no_grad():
        for i in range(NO_y):
            y = y_test[i].clone()
            post_params = get_mixture_posterior(mixture_params, forward_map, b**2, y)
            y = y.repeat(NO_samples, 1)
            z = torch.randn_like(y, device=device)
            for k in range(steps):
                t = torch.ones(NO_samples, 1, device=device) * (k/steps)
                zy = torch.cat((z, y), 1)
                zyt = torch.cat((zy, t), 1)
                z = z - (1/steps) * gen(zyt).detach()
            samples_model = z.clone()
            gt_post = draw_mixture_dist(post_params, NO_samples)
            w2 = Wloss(gt_post, samples_model)
            w2_mean += w2/NO_y
            w_distances.append(w2.item())
            if i < 10:
                make_image(gt_post.cpu().detach().data.numpy(), samples_model.cpu().detach().data.numpy(), 'img_fm_cot'+str(i), directory=directory)
    print(w2_mean)
    return w2_mean, 0

if __name__ == "__main__":
    w_distances = []
    for i in range(3):
            args = parse_arguments()
            args.seed=i
            train_and_eval(args)
    print(np.mean(w_distances))
