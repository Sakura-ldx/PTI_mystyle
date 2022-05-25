import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from configs import paths_config, hyperparameters, global_config
from models.StyleCLIP.criteria import id_loss
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w
from utils.models_utils import toogle_grad, load_old_G


def save_image(img, dir, name):
    img = (img.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
    img = Image.fromarray(img, mode='RGB')
    img.save(os.path.join(dir, f'{name}.jpg'))


class MystyleCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)
        self.id_loss = id_loss.IDLoss().to(global_config.device).eval()

    def train(self):

        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)

        '''use_ball_holder = True
        w_pivots = []
        images = []'''
        # origin latent
        self.G.synthesis.eval()
        self.G.mapping.eval()
        origin_seed = 64
        z_pivot = np.random.RandomState(origin_seed).randn(1, 512)
        with torch.no_grad():
            w_pivot = self.G.mapping(torch.from_numpy(z_pivot).to(global_config.device), None)
            w_pivot = w_pivot.clone().detach()
            image_pivot = self.forward(w_pivot)
            image_pivot = image_pivot.clone().detach()
        torch.save(w_pivot, f'{w_path_dir}/{paths_config.pti_results_keyword}/{origin_seed}.pt')
        dir = f'./result/{global_config.run_name}/origin'
        os.makedirs(dir, exist_ok=True)
        save_image(image_pivot[0], dir, f'{origin_seed}')

        # edit latents
        edit_seed = 123
        z_edits = np.random.RandomState(edit_seed).randn(10, 512)
        w_edits = self.G.mapping(torch.from_numpy(z_edits).to(global_config.device), None)
        w_edits = w_edits.clone().detach()
        w_distances = (w_edits - w_pivot) / torch.norm(w_edits - w_pivot, 2, dim=2, keepdim=True)

        w_finals = []
        a_seed = 347
        a_s = np.random.RandomState(a_seed).uniform(low=-5.0, high=5.0, size=10)
        for a in a_s:
            for w_distance in w_distances:
                w_finals.append(w_pivot + a * w_distance)
        w_finals = torch.cat(w_finals, dim=0)
        print(w_finals.shape)
        batch_size = 4

        for i in tqdm(range(hyperparameters.max_pti_steps)):
            self.G.synthesis.train()
            self.G.mapping.train()

            for j in range(0, w_finals.shape[0], batch_size):
                w_finals_batch = w_finals[j:j + batch_size, :, :]
                real_images_batch = image_pivot.repeat([batch_size, 1, 1, 1])

                generated_images = self.forward(w_finals_batch)
                loss = self.id_loss(generated_images, real_images_batch)
                print(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (i + 1) % 20 == 0:
                self.G.synthesis.eval()
                self.G.mapping.eval()
                result = []
                with torch.no_grad():
                    for j in range(0, w_finals.shape[0], batch_size):
                        w_finals_batch = w_finals[j:j + batch_size, :, :]
                        generated_images = self.forward(w_finals_batch)
                        result.append(generated_images)
                result = torch.cat(result, dim=0)
                dir = './result/{}/train/{:03d}'.format(global_config.run_name, i + 1)
                os.makedirs(dir, exist_ok=True)
                name = 0
                for image in result:
                    save_image(image, dir, '{}_{:03d}'.format(origin_seed, name))
                    name += 1

            global_config.training_step += 1

            # torch.cuda.empty_cache()
        '''if self.use_wandb:
            log_images_from_w(w_pivots, self.G, [image[0] for image in images])'''

        torch.save(self.G,
                   f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_{origin_seed}_{edit_seed}_{a_seed}.pt')
