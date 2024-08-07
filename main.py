import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
from utils import *
import torch
from transformers import DistilBertTokenizer
from CLIP import CLIPModel

from config import CFG
from dataset import *
from modules import *

############Train############
# def make_train_valid_dfs():
#     return train_dataframe, valid_dataframe

def build_loaders(dataset):
    dataset = CustomDataset(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=False,
    )
    return dataloader

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = batch #batch是一个sub-KG
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def main():
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    dataset = Load_dataset(CFG.text_path,CFG.image_path,CFG.motion_path,tokenizer)
    train_loader = build_loaders(dataset)

    model = CLIPModel().to(CFG.device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": model.motion_encoder.parameters(), "lr": CFG.motion_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters(), model.motion_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        if train_loss.avg < best_loss:
            best_loss = train_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")

        lr_scheduler.step(train_loss.avg)


if __name__=='__main__':
    main()