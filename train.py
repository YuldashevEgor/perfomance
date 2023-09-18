import os
import torch
import torch.backends.cudnn as cudnn

from tqdm import tqdm

from datasets import get_dataset_and_dataloader
from models import get_model, get_criterion, get_optimizer, get_scheduler

from utils.tools import get_config, get_device, get_biases_params, clean_device

cudnn.benchmark = True


class Trainer:
    def __init__(self,
                 cfg):

        # Init cfg
        self.cfg = cfg
        self.task_type = cfg["task"]
        # Get device
        self.device = get_device(cfg)
        # Get model and move to device
        model = get_model(cfg["model"][self.task_type], self.task_type)
        # Get criterion
        criterion = get_criterion(cfg['loss'])
        # Initialize the optimizer, with twice the default learning rate for biases
        biases, not_biases = get_biases_params(model)
        self.optimizer = get_optimizer(cfg["optimizer"], biases, not_biases)
        # Get scheduler
        self.scheduler = get_scheduler(cfg["scheduler"], self.optimizer)
        # Init Grad scaler
        self.scaler = torch.cuda.amp.GradScaler()
        # Move to default device
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        # Custom DataLoaders
        self.train_dataset, self.train_loader = get_dataset_and_dataloader("train", cfg["train"])
        self.test_dataset, self.test_loader = get_dataset_and_dataloader("val", cfg["train"])
        # Init training type
        self.fp16 = cfg["train"]["fp16"]
        # Init accumulation grad step
        self.accum_grad_step = cfg["train"]["accum_grad_step"]
        # Init normalize params
        self.mean = torch.as_tensor([0.5, 0.5, 0.5]).view(-1, 1, 1).to(self.device)
        self.std = torch.as_tensor([0.5, 0.5, 0.5]).view(-1, 1, 1).to(self.device)
        # Init num epoch
        self.num_batches = len(self.train_loader)
        self.num_epoch = cfg["train"]["epoch"]
        # Init dtype
        self.dtype = torch.long
        # Init output dir
        self.save_folder = f"./weights/{self.task_type}"
        os.makedirs(self.save_folder, exist_ok=True)

    def train(self):
        # Epochs

        num_iter = 0

        for _ in tqdm(range(1, self.num_epoch + 1),
                      desc=f"Training model ...",
                      leave=True):
            # One epoch's training
            num_iter = self.training_step(num_iter)
            self.scheduler.step()

    def training_step(self, num_iter: int) -> int:
        self.model.train()  # training mode enables dropout

        class_losses = torch.scalar_tensor(0)
        counter = torch.scalar_tensor(0)

        progressbar = tqdm(self.train_loader,
                           desc='Training per epoch...',
                           leave=False)

        # Batches
        for i, (images, targets) in enumerate(progressbar):
            counter += 1
            num_iter += 1
            # Move to default device
            with torch.cuda.amp.autocast(enabled=self.fp16):
                images = images.to(self.device, non_blocking=True)
                # normalization
                images.sub_(self.mean)
                images.div_(self.std)

                # Forward prop.
                predicted_scores = self.model(images)

                del images

                loss = self.criterion(predicted_scores, targets.to(self.device,
                                                                   dtype=self.dtype,
                                                                   non_blocking=True))

            # Backward prop.
            loss /= self.accum_grad_step
            self.scaler.scale(loss).backward()

            if (i + 1) % self.accum_grad_step == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            loss = loss.detach().cpu().item()

            # saving loss value
            class_losses += loss

            progressbar.set_description(f"loss={class_losses / (i + 1):.4f}"),

            # free some memory since their histories may be stored
            del predicted_scores, targets

        del class_losses

        clean_device(self.device)

        return num_iter


if __name__ == '__main__':
    Trainer(cfg=get_config()).train()
