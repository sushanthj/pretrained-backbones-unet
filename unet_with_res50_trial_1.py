# %pip install torch -q
# %pip install opencv-python -q
# %pip install pycocotools -q
# %pip install timm==0.6.12 -q
# %pip install ipdb -q

from backbones_unet.model.unet import Unet
from backbones_unet.utils.dataset import SemanticSegmentationDataset
from backbones_unet.model.losses import DiceLoss
from backbones_unet.utils.trainer import Trainer
from torchsummaryX import summary
from torch.utils.data import Dataset, DataLoader
from convert_coco_ann_to_mask import convert_coco_to_mask
from torchvision.datasets import ImageFolder

import torchvision
import torch
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Test Installation
random_tensor = torch.rand((1, 3, 64, 64))
model = Unet(in_channels=3, num_classes=1) # if no backbone specified, will default to Resnet50
print(model.predict(random_tensor))
# summary(model, random_tensor)

# Feel free to add more items here
config = {
    "lr"         : 1e-4,
    "epochs"     : 100,
    "batch_size" : 1,  # Increase if your device can handle it
    "num_classes": 1,
    'truncated_normal_mean' : 0,
    'truncated_normal_std' : 0.2,
}

# %%
# create a torch.utils.data.Dataset/DataLoader
annotation_json_path = '/home/sush/klab2/rosbags_collated/sensors_2023-08-03-15-19-03_0/annotation.json'
train_img_path = '/home/sush/klab2/rosbags_collated/sensors_2023-08-03-15-19-03_0/images'
train_mask_path = '/home/sush/klab2/rosbags_collated/sensors_2023-08-03-15-19-03_0/masks'

train_img_path_for_ImageFolder_dataloader = '/home/sush/klab2/rosbags_collated/sensors_2023-08-03-15-19-03_0/images_with_class/'

#! Temporarily using train and val images as same
val_img_path = '/home/sush/klab2/rosbags_collated/sensors_2023-08-03-15-19-03_0/images'
val_mask_path = '/home/sush/klab2/rosbags_collated/sensors_2023-08-03-15-19-03_0/masks'

test_img_path = '/home/sush/klab2/rosbags_collated/sensors_2023-08-03-15-19-03_0/images'

# img_size = (1384, 1032) # = width, height            # currently PtGrey images
img_size = (1024, 1024)

# ## Extract Masks from the COCO annotations (if not already done)

# convert_coco_to_mask(input_json=annotation_json_path, image_folder=train_img_path, output_folder=train_mask_path)

# Find mean and std of your dataset:
def get_mean_and_std_calculated(IMAGE_DATA_DIR):
    """
    NOTE: The ImageFolder dataloader requires the following file structure:

    root
    |
    └── cat (class label)
        |
        ├──img_2.png
        └──img_1.png

    """
    train_dataset = ImageFolder(IMAGE_DATA_DIR, transform=torchvision.transforms.ToTensor())

    # Initialize lists to store channel-wise means and standard deviations
    channel_wise_means = [0.0, 0.0, 0.0]
    channel_wise_stds = [0.0, 0.0, 0.0]

    # Iterate through the training dataset to calculate means and standard deviations
    for image, _ in train_dataset:
        for i in range(3):  # Assuming RGB images
            channel_wise_means[i] += image[i, :, :].mean().item()
            channel_wise_stds[i] += image[i, :, :].std().item()

    # Calculate the mean and standard deviation for each channel
    num_samples = len(train_dataset)
    channel_wise_means = [mean / num_samples for mean in channel_wise_means]
    channel_wise_stds = [std / num_samples for std in channel_wise_stds]

    # Print the mean and standard deviation for each channel
    print("Mean:", channel_wise_means)
    print("Std:", channel_wise_stds)

    return channel_wise_means, channel_wise_stds

# means, stds = get_mean_and_std_calculated(train_img_path_for_ImageFolder_dataloader)
means = [0.44895144719250346, 0.4951483853617493, 0.4498602793532975]
stds = [0.21388493326245522, 0.24571933703763144, 0.22413276759337405]

normalize_transform = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(mean=means, std=stds) # always normalize only after tensor conversion
    ])

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(0.5),
    torchvision.transforms.ColorJitter(brightness=0.16, contrast=0.15, saturation=0.1),
    torchvision.transforms.RandomRotation(18),
    torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    torchvision.transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
])

train_dataset = SemanticSegmentationDataset(img_paths=train_img_path, mask_paths=train_mask_path, size=img_size, mode='binary', normalize=normalize_transform, transformations=train_transforms)
val_dataset = SemanticSegmentationDataset(img_paths=val_img_path, mask_paths=val_mask_path, size=img_size, mode='binary', normalize=normalize_transform, transformations=None)
test_dataset = SemanticSegmentationDataset(img_paths=val_img_path, mask_paths=None, size=img_size, normalize=normalize_transform, transformations=None)

temp = train_dataset.__getitem__(1)

# Create data loaders
train_loader = DataLoader(
    dataset     = train_dataset,
    batch_size  = config['batch_size'],
    shuffle     = True,
    num_workers = 4,
    pin_memory  = True
)

val_loader = DataLoader(
    dataset     = val_dataset,
    batch_size  = config['batch_size'],
    shuffle     = False,
    num_workers = 2
)

test_loader = DataLoader(
    dataset     = test_dataset,
    batch_size  = config['batch_size'],
    shuffle     = False,
    drop_last   = False,
    num_workers = 2)

model = Unet(
    # backbone='convnext_base', # backbone network name
    backbone='resnet50',
    preprocessing=True,
    in_channels=3, # input channels (1 for gray-scale images, 3 for RGB, etc.)
    num_classes=config["num_classes"],  # output channels (number of classes in your dataset)
    encoder_freeze=True,
    pretrained=True,
)

# model = model().to(device)
random_tensor = torch.rand((1, 3, 1024, 1024))
print(model.predict(random_tensor))

# Define wandb credentials
import wandb
wandb.login(key="49efd84d0e342f343fb91401332234dea4a3ffe2") #API Key is in your wandb account, under settings (wandb.ai/settings)

run = wandb.init(
    name = "UNet_with_resnet_50", ## Wandb creates random run names if you skip this field
    reinit = True, ### Allows reinitalizing runs when you re-run this cell
    # run_id = ### Insert specific run id here if you want to resume a previous run
    # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
    project = "IDL_Project_Segmentation", ### Project should be created in your wandb account
    config = config ### Wandb Config for your run
)

checkpoint_path = '/home/sush/klab2/Segmentation_Models/checkpoints/checkpoint.pth'
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=config['lr'], betas=(0.9, 0.999), weight_decay=0.05)
gamma = 0.8
milestones = [10,20,40,60,80]

# scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.9, total_iters=5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
# scheduler3 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
# scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2, scheduler3], milestones=[20, 51])

mixed_precision_scaler = torch.cuda.amp.GradScaler()

trainer = Trainer(
    model=model,              # UNet model with Resnet50 backbone
    criterion=DiceLoss(),     # loss function
    optimizer=optimizer,
    epochs=10,
    # scaler=mixed_precision_scaler,
    lr_scheduler=scheduler,
    device=device,
    checkpoint_path=checkpoint_path
)

trainer.fit(train_loader, val_loader)

# Check if the checkpoint file exists
if os.path.exists(checkpoint_path):
    # If the checkpoint file exists, load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']  # last epoch
    val_loss = checkpoint['val_loss']  # Update the best accuracy
    # Load the checkpoint and update the scheduler state if it exists in the checkpoint
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Loaded scheduler state from checkpoint.")
    else:
        print("No scheduler state found in checkpoint.")
    print("Loaded checkpoint from:", checkpoint_path)
else:
    # If the checkpoint file does not exist, start training from scratch
    start_epoch = 0
    print("No checkpoint found at:", checkpoint_path)

print(model)