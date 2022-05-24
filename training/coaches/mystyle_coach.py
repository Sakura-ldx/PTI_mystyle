import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from configs import paths_config, hyperparameters, global_config
from models.StyleCLIP.criteria import id_loss
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w
from utils.models_utils import load_tuned_G, toogle_grad


class MystyleCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)
        self.G = load_tuned_G(global_config.run_name, 'img_16520883136504257_{\'quality\'= 0')
        toogle_grad(self.G, True)
        self.optimizer = self.configure_optimizers()
        self.id_loss = id_loss.IDLoss().to(global_config.device).eval()

    def train(self):
        self.G.synthesis.train()
        self.G.mapping.train()

        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)

        use_ball_holder = True
        w_pivots = []
        images = []

        for fname, image in self.data_loader:
            if self.image_counter >= hyperparameters.max_images_to_invert:
                break

            image_name = fname[0]
            if hyperparameters.first_inv_type == 'w+':
                embedding_dir = f'{w_path_dir}/{paths_config.e4e_results_keyword}/{image_name}'
            else:
                embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
            os.makedirs(embedding_dir, exist_ok=True)

            w_pivot = self.get_inversion(w_path_dir, image_name, image)
            w_pivots.append(w_pivot)
            images.append((image_name, image))
            self.image_counter += 1

        for i in tqdm(range(hyperparameters.max_pti_steps)):
            self.image_counter = 0

            for data, w_pivot in zip(images, w_pivots):
                image_name, image = data

                if self.image_counter >= hyperparameters.max_images_to_invert:
                    break

                real_images_batch = image.to(global_config.device)

                z_edit = torch.randn(5, 512).to(global_config.device)
                w_edit = self.G.mapping(z_edit, None)
                w_edit = w_edit.clone().detach()

                for a in np.linspace(-5, 5, 21):
                    print(a)
                    w_final = w_pivot + a * (w_edit - w_pivot) / torch.norm(w_edit - w_pivot, 2, dim=2, keepdim=True)

                    generated_images = self.forward(w_final)
                    loss = self.id_loss(generated_images, real_images_batch)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    dir = f'./result/{global_config.run_name}/train/{i}/{a}'
                    os.makedirs(dir, exist_ok=True)
                    name = 0
                    for image in generated_images:
                        image = (image.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
                        print(image.shape)
                        image = Image.fromarray(image, mode='RGB')
                        image.save(os.path.join(dir, f'{name}.jpg'))
                        name += 1

                global_config.training_step += 1
                self.image_counter += 1

        if self.use_wandb:
            log_images_from_w(w_pivots, self.G, [image[0] for image in images])

        torch.save(self.G,
                   f'{global_config.run_name}/{paths_config.checkpoints_dir}/model_mystyle.pt')