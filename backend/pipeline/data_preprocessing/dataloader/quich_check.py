from rc_dataloader import RCDataset, collate_pad_validmask, default_mask_resolver, build_transforms_from_cfg, load_yaml_config
from torch.utils.data import DataLoader
import torch
from unet import UNet  # your UNet file
from dataset_generation_scripts.utils import get_config
cfg_mu = get_config("DATA_PREPROC_PATH")

cfg = load_yaml_config("aug_flood.yaml")
transforms = build_transforms_from_cfg(cfg)

ds = RCDataset(
    data_dir=cfg_mu["data_preprocessor_quick_check"]["data_dir"],
    use_masks=True,
    mask_resolver=default_mask_resolver,
    transform=transforms,
    strict_missing_masks=True,
)

loader = DataLoader(
    ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=True,
    collate_fn=lambda b: collate_pad_validmask(b, enforce_stride_multiple=True, stride_multiple=16)
)

batch = next(iter(loader))
print("rc:", batch.rc.shape, "mask:", None if batch.mask is None else batch.mask.shape, "valid:", batch.valid_mask.shape)

model = UNet(in_channels=2, out_channels=1, base_channels=32, depth=4, bilinear=True)
with torch.no_grad():
    logits = model(batch.rc)  # logits at native padded size
print("logits:", logits.shape)
