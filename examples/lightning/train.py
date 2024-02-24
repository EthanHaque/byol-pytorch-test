import os
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader

from byol_pytorch import BYOL
import pytorch_lightning as pl

from torchvision import datasets

BATCH_SIZE = 256
EPOCHS = 2
LR = 3e-4
NUM_GPUS = int(os.environ["SLURM_GPUS_ON_NODE"])
IMAGE_SIZE = 224
NUM_WORKERS = int(os.environ['SLURM_CPUS_PER_TASK'])


class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images):
        return self.learner(images)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        loss = self.forward(images)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()


class PredictWrapper(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = net

    def forward(self, images):
        return self.learner(images)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images, labels = batch
        return self.forward(images), labels


def setup_imagenet_datasets(root):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    ds_train = datasets.ImageNet(root=root, split='train', transform=transform)
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    ds_test = datasets.ImageNet(root=root, split='val', transform=transform)
    ds_test = torch.utils.data.Subset(ds_test, list(range(500)))
    test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    return train_loader, test_loader


if __name__ == '__main__':
    resnet = models.resnet50(pretrained=True)

    train_loader, test_loader = setup_imagenet_datasets(
        '/scratch/gpfs/DATASETS/imagenet/ilsvrc_2012_classification_localization')

    model = SelfSupervisedLearner(
        resnet,
        image_size=IMAGE_SIZE,
        hidden_layer='avgpool',
        projection_size=256,
        projection_hidden_size=4096,
        moving_average_decay=0.99
    )
    trainer = pl.Trainer(
        gpus=NUM_GPUS,
        max_epochs=EPOCHS,
        accumulate_grad_batches=1,
        sync_batchnorm=True
    )

    trainer.fit(model, train_loader)
