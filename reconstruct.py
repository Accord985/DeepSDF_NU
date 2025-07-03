#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import random
import time
import torch

import deep_sdf
import deep_sdf.workspace as ws

import numpy as np

def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    stat,
    clamp_dist,
    num_samples=30000,
    lr=5e-4,
    l2reg=False,
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()

    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss()

    num_samp_per_scene = specs["SamplesPerScene"]
    # scene_per_batch = specs["ScenesPerBatch"]
    batch_split = 1  # a constant that appears in train_deep_sdf_positional_encoding. We never used that value and its default is 1.
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True

    for e in range(num_iterations):

        decoder.eval()
        sdf_data = deep_sdf.data.unpack_sdf_samples_from_ram(
            test_sdf, num_samples
        ).cuda()
        
        xyz = sdf_data[:, 0:3]

        L = 10
        xyz_el = []

        for el in range(0,L):
            val = 2 ** el

            x = np.sin(val * np.pi * xyz[:, 0].cpu().numpy())
            xyz_el.append(x)
            x = np.cos(val * np.pi * xyz[:, 0].cpu().numpy())
            xyz_el.append(x)
            y = np.sin(val * np.pi * xyz[:, 1].cpu().numpy())
            xyz_el.append(y)
            y = np.cos(val * np.pi * xyz[:, 1].cpu().numpy())
            xyz_el.append(y)
            z = np.sin(val * np.pi * xyz[:, 2].cpu().numpy())
            xyz_el.append(z)
            z = np.cos(val * np.pi * xyz[:, 2].cpu().numpy()) 
            xyz_el.append(z)

        xyz_el = np.array(xyz_el)
        xyz_el = torch.tensor(xyz_el, dtype=torch.float32).T
        #xyz_el = torch.tensor(xyz_el).T
        xyz = xyz_el
        sdf_gt = sdf_data[:, 3].unsqueeze(1)

        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)

        inputs = torch.cat([latent_inputs.cuda(), xyz.cuda()], 1)

        pred_sdf = decoder(inputs)

        # TODO: why is this needed?
        if e == 0:
            pred_sdf = decoder(inputs)

        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        loss = loss_l1(pred_sdf, sdf_gt)    
        if l2reg:
            loss += 1e-4 * torch.mean(latent.pow(2))
        loss.backward()
        optimizer.step()

        if e % 50 == 0:
            logging.debug(loss.cpu().data.numpy())
            logging.debug(e)
            logging.debug(latent.norm())
        loss_num = loss.cpu().data.numpy()

    # means_list = [0.005520996, -0.012547219, -0.004975365, 0.001681339, -0.026489312, 0.011676402, 0.02658528, 0.029440958, -0.003096161, -0.006701054, 0.021360265, -0.006143919, -0.02814932, 0.012072136, -0.010563934, -0.030706026, 0.0070554, 0.008844139, -0.005916632, 0.012540568, -0.011084815, -0.030274, 0.022648365, -0.015611669, -0.020734984, 0.004469838, 0.019983489, -0.013408898, 0.015596918, -0.002954856, 0.003465294, 0.012333186, 0.026264839, 0.0252841, 0.013649611, 0.009940355, -0.007092595, 0.013736133, 0.015240351, -0.007368188] # 平均値
    # if len(means_list) < latent_size:
    #     means_list.extend([0.0] * (latent_size - len(means_list))) # 足りない場合は0で埋める (必要に応じて変更)
    # means = torch.tensor(means_list).float()

    # stds_list = [0.137812604, 0.127337706, 0.166447835, 0.125315042, 0.150552954, 0.131581895, 0.131364864, 0.147621291, 0.137656617, 0.145154149, 0.122065383, 0.135400064, 0.12298796, 0.144837127, 0.145894223, 0.152496485, 0.159087613, 0.146982587, 0.134903452, 0.153211536, 0.145440148, 0.130786875, 0.158340973, 0.136564079, 0.13627438, 0.138197789, 0.140747448, 0.139792739, 0.131838043, 0.12880111, 0.141944621, 0.141399522, 0.151067955, 0.130762302, 0.132840619, 0.131526149, 0.144109777, 0.158203106, 0.135004588, 0.143835015] # 標準偏差
    # if len(stds_list) < latent_size:
    #     stds_list.extend([1.0] * (latent_size - len(stds_list))) # 足りない場合は1で埋める (必要に応じて変更)
    # stds = torch.tensor(stds_list).float()

    # latent = torch.zeros(1, latent_size).cuda()

    # for i in range(latent_size):
    #     latent[0, i] = torch.normal(mean=means[i], std=stds[i])

    return loss_num, latent


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to reconstruct.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=800,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()

    with open(args.split_filename, "r") as f:
        split = json.load(f)

    npz_filenames = deep_sdf.data.get_instance_filenames(args.data_source, split)

    random.shuffle(npz_filenames)

    logging.debug(decoder)

    err_sum = 0.0
    repeat = 1
    save_latvec_only = False
    rerun = 0

    reconstruction_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, str(saved_model_epoch)
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)

    for ii, npz in enumerate(npz_filenames):

        if "npz" not in npz:
            continue

        full_filename = os.path.join(args.data_source, ws.sdf_samples_subdir, npz)

        logging.debug("loading {}".format(npz))

        data_sdf = deep_sdf.data.read_sdf_samples_into_ram(full_filename)

        for k in range(repeat):

            if rerun > 1:
                mesh_filename = os.path.join(
                    reconstruction_meshes_dir, npz[:-4] + "-" + str(k + rerun)
                )
                latent_filename = os.path.join(
                    reconstruction_codes_dir, npz[:-4] + "-" + str(k + rerun) + ".pth"
                )
            else:
                mesh_filename = os.path.join(reconstruction_meshes_dir, npz[:-4])
                latent_filename = os.path.join(
                    reconstruction_codes_dir, npz[:-4] + ".pth"
                )

            if (
                args.skip
                and os.path.isfile(mesh_filename + ".ply")
                and os.path.isfile(latent_filename)
            ):
                continue

            logging.info("reconstructing {}".format(npz))

            data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
            data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

            start = time.time()
            err, latent = reconstruct(
                decoder,
                int(args.iterations),
                latent_size,
                data_sdf,
                0.01,  # [emp_mean,emp_var],
                0.1,
                num_samples=8000,
                lr=5e-3,
                l2reg=True,
            )
            logging.debug("reconstruct time: {}".format(time.time() - start))
            err_sum += err
            logging.debug("current_error avg: {}".format((err_sum / (ii + 1))))
            logging.debug(ii)

            logging.debug("latent: {}".format(latent.detach().cpu().numpy()))

            decoder.eval()

            if not os.path.exists(os.path.dirname(mesh_filename)):
                os.makedirs(os.path.dirname(mesh_filename))

            if not save_latvec_only:
                start = time.time()
                with torch.no_grad():
                    deep_sdf.mesh.create_mesh(
                        decoder, latent, mesh_filename, N=256, max_batch=int(2 ** 18)
                    )
                logging.debug("total time: {}".format(time.time() - start))

            if not os.path.exists(os.path.dirname(latent_filename)):
                os.makedirs(os.path.dirname(latent_filename))

            torch.save(latent.unsqueeze(0), latent_filename)
