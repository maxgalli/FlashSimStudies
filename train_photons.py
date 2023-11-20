import hydra
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import os
import sys
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pickle as pkl
import corner

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group


# where wipfs modules are
sys.path.append("/work/gallim/SIMStudies/wipfs/models")
sys.path.append("/work/gallim/SIMStudies/FlashSimStudies/preprocessing")

# seed
np.random.seed(42)

from modded_basic_nflow import create_mixture_flow_model, load_mixture_model, save_model
from utils.log import setup_comet_logger
from utils.performance import EarlyStopping
from utils.datasets import PhotonDataset
from utils.plots import sample_and_plot


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def train(device, cfg, world_size=None, device_ids=None):
    # device is device when not distributed and rank when distributed

    if world_size is not None:
        ddp_setup(device, world_size)

    device_id = device_ids[device] if device_ids is not None else device

    # create dict as required by the function
    input_dim = len(cfg.target_variables)
    context_dim = len(cfg.context_variables)
    flow_params_dct = {
        "input_dim": input_dim,
        "context_dim": context_dim,
        "base_kwargs": {
            "num_steps_maf": cfg.model.maf.num_steps,
            "num_steps_arqs": cfg.model.arqs.num_steps,
            "num_transform_blocks_maf": cfg.model.maf.num_transform_blocks,
            "num_transform_blocks_arqs": cfg.model.arqs.num_transform_blocks,
            "activation": cfg.model.activation,
            "dropout_probability_maf": cfg.model.maf.dropout_probability,
            "dropout_probability_arqs": cfg.model.arqs.dropout_probability,
            "use_residual_blocks_maf": cfg.model.maf.use_residual_blocks,
            "use_residual_blocks_arqs": cfg.model.arqs.use_residual_blocks,
            "batch_norm_maf": cfg.model.maf.batch_norm,
            "batch_norm_arqs": cfg.model.arqs.batch_norm,
            "num_bins_arqs": cfg.model.arqs.num_bins,
            "tail_bound_arqs": cfg.model.arqs.tail_bound,
            "hidden_dim_maf": cfg.model.maf.hidden_dim,
            "hidden_dim_arqs": cfg.model.arqs.hidden_dim,
            "init_identity": cfg.model.init_identity,
        },
        "transform_type": cfg.model.transform_type,
    }
    model = create_mixture_flow_model(**flow_params_dct).to(device)
    if cfg.checkpoint is not None:
        # assume that the checkpoint is path to a directory
        model, _, _, start_epoch, th, _ = load_mixture_model(
            model, model_dir=cfg.checkpoint, filename="checkpoint-latest.pt"
        )
        model = model.to(device)
        best_train_loss = np.min(th)
        print("Loaded model from checkpoint: ", cfg.checkpoint)
        print("Resuming from epoch ", start_epoch)
        print("Best train loss found to be: ", best_train_loss)
    else:
        start_epoch = 1
        best_train_loss = 10000000
    # print(f"Memory allocated on device {device_id} after creating model: {readable_allocated_memory(torch.cuda.memory_allocated(device))}")

    if world_size is not None:
        ddp_model = DDP(
            model,
            device_ids=[device],
            output_device=device,
            #find_unused_parameters=True,
        )
        model = ddp_model.module
    else:
        ddp_model = model
    # print(f"Memory allocated on device {device_id} after calling DDP: {readable_allocated_memory(torch.cuda.memory_allocated(device))}")
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    early_stopping = EarlyStopping(
        patience=cfg.stopper.patience, min_delta=cfg.stopper.min_delta
    )

    # make datasets
    train_dataset = PhotonDataset(
        cfg.train.path,
        cfg.context_variables,
        cfg.target_variables,
        device=device,
        rows=cfg.train.size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False if world_size is not None else True,
        sampler=DistributedSampler(train_dataset) if world_size is not None else None,
        # num_workers=2,
        #pin_memory=True,
    )
    test_dataset = PhotonDataset(
        cfg.test.path,
        cfg.context_variables,
        cfg.target_variables,
        device=device,
        rows=cfg.test.size,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        # num_workers=2,
        #pin_memory=True,
    )
    # print(f"Memory allocated on device {device_id} after creating datasets: {readable_allocated_memory(torch.cuda.memory_allocated(device))}")

    # train the model
    writer = SummaryWriter(log_dir="runs")

    # setup comet logger
    comet_name = os.getcwd().split("/")[-1]
    comet_logger = setup_comet_logger(comet_name, cfg)

    optimizer = torch.optim.Adam(
        # optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.optimizer.learning_rate,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)

    train_history, test_history = [], []
    for epoch in range(start_epoch, cfg.epochs + 1):
        if world_size is not None:
            b_sz = len(next(iter(train_loader))[0])
            print(
                f"[GPU{device_id}] | Rank {device} | Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(train_loader)}"
            )
            train_loader.sampler.set_epoch(epoch)
        print(f"Epoch {epoch}/{cfg.epochs}:")
        train_losses = []
        test_losses = []
        # train
        start = time.time()
        print("Training...")
        for i, (context, target) in enumerate(train_loader):
            context, target = context.to(device), target.to(device)
            # context, target = context.to(dtype=torch.float16).to(device), target.to(dtype=torch.float16).to(device)
            # print(target.shape)
            # print(f"Memory on device {device_id} after moving data of batch {i}: {readable_allocated_memory(torch.cuda.memory_allocated(device))}")
            model.train()
            optimizer.zero_grad()
            # for param in ddp_model.module.parameters():
            #    param.grad = None

            # with profile(
            #    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            #    record_shapes=True,
            # ) as prof:
            #    with record_function("forward pass"):
            log_prog, logabsdet = ddp_model(target, context=context)
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
            # print(f"Memory on device {device_id} after forward pass of batch {i}: {readable_allocated_memory(torch.cuda.memory_allocated(device))}")
            loss = -log_prog - logabsdet
            loss = loss.mean()
            train_losses.append(loss.item())

            loss.backward()
            # print(f"Memory on device {device_id} after backward pass of batch {i}: {readable_allocated_memory(torch.cuda.memory_allocated(device))}")
            optimizer.step()
            scheduler.step()
            # print(f"Memory on device {device_id} after optimizer step of batch {i}: {readable_allocated_memory(torch.cuda.memory_allocated(device))}")

        epoch_train_loss = np.mean(train_losses)
        # if world_size is not None:
        #    epoch_train_loss *= world_size
        train_history.append(epoch_train_loss)

        # test
        print("Testing...")
        for i, (context, target) in enumerate(test_loader):
            context, target = context.to(device), target.to(device)
            with torch.no_grad():
                model.eval()
                log_prog, logabsdet = ddp_model(target, context=context)
                loss = -log_prog - logabsdet
                loss = loss.mean()
                test_losses.append(loss.item())
        # print(f"Memory allocated on device {device_id} after testing: {readable_allocated_memory(torch.cuda.memory_allocated(device))}")

        epoch_test_loss = np.mean(test_losses)
        # if world_size is not None:
        #    epoch_test_loss *= world_size
        test_history.append(epoch_test_loss)
        if device == 0 or world_size is None:
            writer.add_scalars(
                "Losses", {"train": epoch_train_loss, "val": epoch_test_loss}, epoch
            )
            comet_logger.log_metrics(
                {"train": epoch_train_loss, "val": epoch_test_loss}, step=epoch
            )

        # sample and validation
        if epoch % cfg.sample_every == 0 or epoch == 1:
            print("Sampling and plotting...")
            sample_and_plot(
                test_loader=test_loader,
                model=model,
                epoch=epoch,
                writer=writer,
                comet_logger=comet_logger,
                context_variables=cfg.context_variables,
                target_variables=cfg.target_variables,
                device=device,
            )

        duration = time.time() - start
        print(
            f"Epoch {epoch} | GPU{device_id} | Rank {device} - train loss: {epoch_train_loss:.4f} - val loss: {epoch_test_loss:.4f} - time: {duration:.2f}s"
        )

        if device == 0 or world_size is None:
            save_model(
                epoch,
                ddp_model,
                scheduler,
                train_history,
                test_history,
                name="checkpoint-latest.pt",
                model_dir=".",
                optimizer=optimizer,
                is_ddp=world_size is not None,
            )
        
        if epoch_train_loss < best_train_loss:
            print("New best train loss, saving model...")
            best_train_loss = epoch_train_loss
            if device == 0 or world_size is None:
                save_model(
                    epoch,
                    ddp_model,
                    scheduler,
                    train_history,
                    test_history,
                    name="best_train_loss.pt",
                    model_dir=".",
                    optimizer=optimizer,
                    is_ddp=world_size is not None,
                )

        early_stopping(epoch_train_loss)
        if early_stopping.early_stop:
            break

    writer.close()


@hydra.main(version_base=None, config_path="config_photons", config_name="cfg0")
def main(cfg):
    # This because in the hydra config we enable the feature that changes the cwd to the experiment dir
    initial_dir = get_original_cwd()
    print("Initial dir: ", initial_dir)
    print("Current dir: ", os.getcwd())

    # save the config
    print(cfg)
    cfg_name = HydraConfig.get().job.name
    with open(f"{os.getcwd()}/{cfg_name}.yaml", "w") as file:
        OmegaConf.save(config=cfg, f=file)

    env_var = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_var:
        actual_devices = env_var.split(",")
        actual_devices = [int(d) for d in actual_devices]
    else:
        actual_devices = list(range(torch.cuda.device_count()))
    print("Actual devices: ", actual_devices)

    print("Training with cfg: \n", OmegaConf.to_yaml(cfg))
    if cfg.distributed:
        world_size = torch.cuda.device_count()
        # make a dictionary with k: rank, v: actual device
        dev_dct = {i: actual_devices[i] for i in range(world_size)}
        print(f"Devices dict: {dev_dct}")
        mp.spawn(
            train,
            args=(cfg, world_size, dev_dct),
            nprocs=world_size,
            join=True,
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train(device, cfg)


if __name__ == "__main__":
    main()
