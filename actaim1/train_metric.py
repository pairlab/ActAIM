import argparse
from pathlib import Path
from datetime import datetime
import torch
from torch.utils import tensorboard
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pdb

from affordance.metric.dataset_metric import DatasetMetric, DatasetModesMetric
from affordance.metric.encoder import VoxelEncoder, PixelEncoder, DepthEncoder, AutoEncoder, Classifier
from affordance.metric.loss import MarginLoss, LogRatioLoss


def save_training_plot(x, y, name):
    plt.plot(x, y)
    plt.xlabel("Iteration")
    plt.ylabel(name + " Loss")
    training_plot_name = name + "_train_plot.png"
    plt.savefig(training_plot_name)
    plt.clf()


def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    num_workers = 1
    batch_size = 1

    # TODO this is the real batch size
    _batch_size = 16

    if args.savedir == "":
        # create log directory
        time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M")
        description = "metric,time={},dim={},obs={},loss={}net={},batch_size={},lr={:.0e},{}".format(
            time_stamp,
            args.dim,
            args.obs,
            args.loss,
            args.net,
            args.batch_size,
            args.lr,
            args.description,
        ).strip(",")
        logdir = args.logdir / description
    else:
        logdir = Path(args.savedir)

    logdir.mkdir(parents=True, exist_ok=True)

    # create data loaders
    dataset = DatasetModesMetric(args.obs, batch_size) if args.modes else DatasetMetric(args.obs, batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=_batch_size, shuffle=True, num_workers=num_workers)
    # test_dataset = DatasetModesMetric(args.obs, batch_size) if args.modes else DatasetMetric(args.obs, batch_size)
    # testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=None, shuffle=True, num_workers=num_workers)

    # create metric learning encoder
    if args.obs == "tsdf":
        model = VoxelEncoder().to(device)
    elif args.obs == "rgbd":
        model = PixelEncoder().to(device)
    elif args.obs == "depth":
        model = DepthEncoder().to(device)

    if args.net == "autoencoder":
        model = AutoEncoder(c_dim=args.dim).to(device)
        # path = "./data/metric/metric,time=22-01-17-16-05,obs=depth,loss=marginnet=autoencoder,batch_size=32,lr=1e-06/metric.pt"
        # path = './model/metric.pt'
        # model.load_state_dict(torch.load(path))
    elif args.net == "autoencoder_class":
        model = AutoEncoder().to(device)
        classifier = Classifier().to(device)

        # TODO load from trained model
        path = "./data/metric/metric,time=22-01-17-16-05,obs=depth,loss=marginnet=autoencoder,batch_size=32,lr=1e-06/metric.pt"
        # path = './model/metric.pt'
        model.load_state_dict(torch.load(path))

    # define optimizer and metrics
    # TODO only train the classifier
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam([
    #     {'params': model.parameters()},
    #     {'params': classifier.parameters()}
    # ], lr=args.lr)

    # define loss, using cos similarity
    criterion = None
    if args.loss == "margin":
        criterion = MarginLoss(0.4, args.modes)
    elif args.loss == "logratio":
        criterion = LogRatioLoss()
    else:
        print("Invalid loss method!")
        exit()
    if args.net == "autoencoder":
        criterion = torch.nn.MSELoss()
    elif args.net == "autoencoder_class":
        criterion = torch.nn.MSELoss()
        class_criterion = torch.nn.BCELoss()

    loss_values = []
    epoch_values = []
    for epoch in range(250):
        print("---epoch: ", epoch)
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0
        for i, data in enumerate(dataloader):
            obs, dof_dist, init_obs = prepare_data(data, device, args.modes)
            if args.modes and args.net == "metric":
                encode_z = model(obs)
                encode_init = model(init_obs)
                encode_init_z = encode_init.detach()
                encode_z = encode_z - encode_init_z
            if args.net == "autoencoder":
                encode_z = model(obs.squeeze(1))
                loss = criterion(obs, encode_z)
            elif args.net == "autoencoder_class":
                # TODO manually reset the shape
                obs = obs.squeeze()
                init_obs = init_obs.squeeze()

                encode_z = model.encode(obs)
                encode_init = model.encode(init_obs)
                reconstruct = model.decode(encode_z)

                reconstruct_loss = criterion(obs, reconstruct)
                # encode_init_repeat = encode_init.repeat((batch_size, 1))
                class_predict = classifier(encode_init, encode_z)
                class_loss = class_criterion(class_predict, dof_dist)
                loss = class_loss

            else:
                loss = criterion.forward(encode_z, dof_dist)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        loss_values.append(epoch_loss)
        optimizer.zero_grad()
        print("---avg_loss: ", epoch_loss)
        epoch_values.append(epoch)

        if (epoch + 0) % (10) == 0:
            model_name = "metric_" + str(args.dim) + ".pt"
            save_model(model, logdir, model_name)
            if args.net == "autoencoder_class":
                save_model(classifier, logdir, "classifier.pt")
            save_training_plot(epoch_values, loss_values, args.net)


def save_model(model, save_dir, model_name):
    filename = save_dir / model_name
    torch.save(model.state_dict(), str(filename))


def prepare_data(data, device, modes):
    if modes:
        obs, dof_dist, init_obs = data
        init_obs = init_obs.float().unsqueeze(0).to(device)
    else:
        obs, dof_dist = data
        init_obs = None
    obs = obs.float().to(device)
    dof_dist = dof_dist.float().to(device)

    return obs, dof_dist, init_obs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", default="autoencoder")
    parser.add_argument("--logdir", type=Path, default="data/metric")
    parser.add_argument("--description", type=str, default="classification_loss_only")
    parser.add_argument("--savedir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument("--modes", type=bool, default=True)
    parser.add_argument("--dim", type=int, default=128)

    # Train with margin loss or log ratio loss
    parser.add_argument("--loss", type=str, default="margin")

    # Train with rgbd as obs or tsdf as obs
    parser.add_argument("--obs", type=str, default="depth")
    args = parser.parse_args()
    print(args)
    main(args)
