import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from configs import paths_config, hyperparameters, global_config
from editings.sefa import factorize_weight_stylegan3
from models.StyleCLIP.criteria import id_loss
from training.coaches.base_coach import BaseCoach


def save_image(img, dir, name):
    img = (img.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
    img = Image.fromarray(img, mode='RGB')
    img.save(os.path.join(dir, f'{name}.jpg'))


class MystyleCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)
        self.id_loss = id_loss.IDLoss().to(global_config.device).eval()
        self.origin_seed = 64
        self.edit_seed = 123
        self.a_seed = 347
        self.batch_size = 4
        self.random = False

    def train(self):

        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)
        images_path_dir = './result'
        os.makedirs(f'{images_path_dir}', exist_ok=True)
        if self.random:
            g_path = f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_{self.origin_seed}_{self.edit_seed}_{self.a_seed}.pt'
        else:
            g_path = f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_{self.origin_seed}_sefa.pt'

        w_pivot, image_pivot = self.generate_origin(self.origin_seed, w_path_dir, images_path_dir)

        for i in tqdm(range(hyperparameters.max_pti_steps)):
            w_finals = self.generate_edits(w_pivot)
            if i == 0:
                self.validate(w_finals, i, images_path_dir)

            self.G.synthesis.train()
            self.G.mapping.train()
            real_images_batch = image_pivot.repeat([self.batch_size, 1, 1, 1])

            generated_images = self.forward(w_finals)
            loss = self.id_loss(generated_images, real_images_batch)
            print(loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (i + 1) % 20 == 0:
                self.validate(w_finals, i + 1, images_path_dir)

            global_config.training_step += 1

        '''if self.use_wandb:
            log_images_from_w(w_pivots, self.G, [image[0] for image in images])'''

        torch.save(self.G, g_path)

    def validate(self, w_finals, epoch, images_path_dir):
        if self.random:
            dir = '{}/{}/train_{}_{}/{:03d}'.format(images_path_dir, global_config.run_name,
                                                    self.edit_seed, self.a_seed, epoch)
        else:
            dir = '{}/{}/train_sefa/{:03d}'.format(images_path_dir, global_config.run_name, epoch)
        os.makedirs(dir, exist_ok=True)

        self.G.synthesis.eval()
        self.G.mapping.eval()
        with torch.no_grad():
            generated_images = self.forward(w_finals)

        name = 0
        for image in generated_images:
            save_image(image, dir, '{}_{:03d}'.format(self.origin_seed, name))
            name += 1

    def generate_origin(self, origin_seed, w_path_dir, images_path_dir):
        dir = f'{images_path_dir}/{global_config.run_name}/origin'
        os.makedirs(dir, exist_ok=True)

        self.G.synthesis.eval()
        self.G.mapping.eval()
        z_pivot = np.random.RandomState(origin_seed).randn(1, 512)
        with torch.no_grad():
            w_pivot = self.G.mapping(torch.from_numpy(z_pivot).to(global_config.device), None)
            w_pivot = w_pivot.clone().detach()
            image_pivot = self.forward(w_pivot)
            image_pivot = image_pivot.clone().detach()
        torch.save(w_pivot, f'{w_path_dir}/{paths_config.pti_results_keyword}/{origin_seed}.pt')

        save_image(image_pivot[0], dir, f'{origin_seed}')
        return w_pivot, image_pivot

    def generate_edits(self, w_pivot):
        if self.random:
            z_edits = np.random.RandomState(self.edit_seed).randn(10, 512)
            with torch.no_grad():
                w_edits = self.G.mapping(torch.from_numpy(z_edits).to(global_config.device), None)
            w_edits = w_edits.clone().detach()
            w_edits = (w_edits - w_pivot) / torch.norm(w_edits - w_pivot, 2, dim=2, keepdim=True)
            w_edits = w_edits[:, 0, :]
        else:
            boundaries, _ = factorize_weight_stylegan3(self.G)
            w_edits = boundaries[:32]

        weights = np.random.RandomState(478).randn(self.batch_size, w_edits.shape[0])
        w_edits = np.matmul(weights, w_edits)
        w_edits = w_edits / np.linalg.norm(w_edits, 2, axis=1, keepdims=True)
        a_s = np.random.RandomState(self.a_seed).uniform(low=-5.0, high=5.0, size=(self.batch_size, 1))

        w_finals = w_pivot + torch.from_numpy(a_s * w_edits).unsqueeze(1).repeat([1, 18, 1]).float().to(global_config.device)
        return w_finals


