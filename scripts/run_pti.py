import sys
sys.path.append(".")
sys.path.append("..")

from random import choice
from string import ascii_uppercase
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os
from configs import global_config, paths_config, hyperparameters
import wandb
from argparse import ArgumentParser

from training.coaches.mystyle_coach import MystyleCoach
from training.coaches.multi_id_coach import MultiIDCoach
from training.coaches.single_id_coach import SingleIDCoach
from utils.ImagesDataset import ImagesDataset


def run_PTI(run_name='', use_wandb=False, use_multi_id_training=False, mystyle=False):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    if run_name == '':
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    else:
        global_config.run_name = run_name

    if use_wandb:
        run = wandb.init(project=paths_config.pti_results_keyword, reinit=True, name=global_config.run_name)
    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    embedding_dir_path = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}/{paths_config.pti_results_keyword}'
    os.makedirs(embedding_dir_path, exist_ok=True)

    dataset = ImagesDataset(paths_config.input_data_path, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    if not hyperparameters.max_images_to_invert:
        hyperparameters.max_images_to_invert = len(dataloader)

    if use_multi_id_training:
        coach = MultiIDCoach(dataloader, use_wandb)
    elif mystyle:
        coach = MystyleCoach(dataloader, use_wandb)
    else:
        coach = SingleIDCoach(dataloader, use_wandb)

    coach.train()

    return global_config.run_name


def options():
    parser = ArgumentParser()
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints')
    parser.add_argument('--embedding_base_dir', type=str, default='embeddings')
    parser.add_argument('--input_data_path', type=str)

    parser.add_argument('--max_images_to_invert', default=None)
    parser.add_argument('--first_inv_type', type=str, default='w')
    parser.add_argument('--max_pti_steps', type=int, default=350)
    parser.add_argument('--first_inv_steps', type=int, default=450)
    parser.add_argument('--use_last_w_pivots', type=bool, default=True)

    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--use_wandb', type=bool, default=False)
    parser.add_argument('--use_multi_id_training', type=bool, default=False)

    opts = parser.parse_args()
    return opts


if __name__ == '__main__':
    opts = options()
    paths_config.checkpoints_dir = opts.checkpoints_dir
    paths_config.embedding_base_dir = opts.embedding_base_dir
    paths_config.input_data_path = opts.input_data_path
    paths_config.input_data_id = os.path.basename(paths_config.input_data_path)

    hyperparameters.max_images_to_invert = opts.max_images_to_invert
    hyperparameters.first_inv_type = opts.first_inv_type
    hyperparameters.max_pti_steps = opts.max_pti_steps
    hyperparameters.first_inv_steps = opts.first_inv_steps
    hyperparameters.use_last_w_pivots = opts.use_last_w_pivots

    run_PTI(run_name=opts.run_name, use_wandb=opts.use_wandb,
            use_multi_id_training=opts.use_multi_id_training, mystyle=True)
