
import yaml
import torch.multiprocessing as mp
from toolkit.init import *


# TODO figure issue with performance on multiple gpu
# TODO Add data prep to config

# TODO Test MLP ratios
# TODO Test window sizes

# do downscale when 0-255
import wandb

def training(rank, config, world_size=0):
    model, helper, (train_dataset, test_dataset, sampler, train_loader, test_loader) = init(config, rank, world_size)
    if rank == 0:
        # Start W&B run
        if config["logging"]["wandb_log"]:
            wandb.login()
            wandb.init(
                project="SuperResolution",
                config=config
            )
            run_name = wandb.run.name
            config["logging"]["model_save_name"] = run_name
        print("Running Training...")

    helper.train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        train_dataset=train_dataset,
        test_dataset=test_dataset
    )
    if world_size != 0:
        dist.destroy_process_group()
    if config["logging"]["wandb_log"]:
        wandb.finish()

def sample_images(config, count):
    model, helper, (train_dataset, test_dataset, train_loader, test_loader) = init(config)
    print("Running Image Sampling...")
    helper.sample_model(random_sample=count, dataset=train_dataset, save_img=True)

def test(rank, config, world_size=0):
    model, helper, (train_dataset, test_dataset, sampler, train_loader, test_loader) = init(config, rank, world_size)
    #from LMDB import read_from_lmdb, get_keys

    epoch_validation_losses, ssim, psnr = helper.validation_loop(test_loader, 10, 10, 20, interp=False)
    print(f"{np.mean(psnr):.4g} / {np.mean(ssim):.4g}")


if __name__ == '__main__':
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    #test(config=config, rank=0, world_size=0)

    # Training
    if config["tools"]["multi_gpu_enable"]:
        world_size = torch.cuda.device_count()
        mp.spawn(training, args=(config, world_size,), nprocs=world_size, join=True)
    else:
        training(config=config, rank=0, world_size=0)



    #training(config)

    #sample_images(config, 1)

'''
datasets Y
file_backend Y
init Y
lmdb_utils Y
loss_function Y
model_helper N
param_ema Y
sr_pairs Y
subimg Y
transofmrs Y
utils N
data prep N
main N



'''