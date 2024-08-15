import itertools
from tqdm.autonotebook import tqdm
from utils import *
import torch
from transformers import DistilBertTokenizer, AutoTokenizer
from config import CFG
from dataset import *
from modules import *
import random
import deepspeed
from CLIP import CLIPModel


def build_loaders(dataset):
    dataset = CustomDataset(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=False,
    )
    return dataloader


def to_device(batch, device):
    for key, value in batch.items():
        if key == 'text':
            batch[key] = [
                [{k: v.to(device) for k, v in item.items()} for item in text_list]
                for text_list in value
            ]
        elif key in ['image', 'motion']:
            batch[key] = [tensor.to(device) for tensor in value]
    return batch


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    len_train_loader = len(train_loader)
    tqdm_object = tqdm(train_loader, total=len_train_loader)
    for idx, batch in enumerate(tqdm_object):

        random_idx = random.choice([i for i in range(len_train_loader - 1) if i != idx])
        batch_prime = None
        for i, data in enumerate(train_loader):
            if i == random_idx:
                batch_prime = data
                break
        batch = to_device(batch, CFG.device)
        batch_prime = to_device(batch_prime, CFG.device)
        loss = model(batch, batch_prime)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"][0].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def train_epoch_deepspeed(model_engine, train_loader):
    loss_meter = AvgMeter()
    len_train_loader = len(train_loader)
    tqdm_object = tqdm(train_loader, total=len_train_loader)
    for idx, batch in enumerate(tqdm_object):
        random_idx = random.choice([i for i in range(len_train_loader - 1) if i != idx])
        batch_prime = None
        for i, data in enumerate(train_loader):
            if i == random_idx:
                batch_prime = data
                break
        batch_prime = to_device(batch_prime, CFG.device)
        batch = to_device(batch, CFG.device)
        loss = model_engine(batch, batch_prime)

        model_engine.backward(loss)
        model_engine.step()
        count = batch["image"][0].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(model_engine.optimizer))

    return loss_meter



def main():
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    dataset = Load_dataset(CFG.text_path, CFG.image_path, CFG.motion_path, tokenizer)
    train_loader = build_loaders(dataset)
    model = CLIPModel()

    if CFG.deepspeed:
        params = [
            {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
            {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
            {"params": model.motion_encoder.parameters(), "lr": CFG.motion_encoder_lr},
            {"params": itertools.chain(
                model.image_projection.parameters(), model.text_projection.parameters(),
                model.motion_projection.parameters()
            ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
        ]
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=params,
            config="ds_config.json"
        )

        best_loss = float('inf')
        for epoch in range(CFG.epochs):
            print(f"Epoch: {epoch + 1}")
            model_engine.train()
            train_loss = train_epoch_deepspeed(model_engine, train_loader)
            print('loss:', train_loss)
            if train_loss.avg < best_loss:
                best_loss = train_loss.avg
                model_engine.save_checkpoint("final_model")
                print("Finish")

    else:
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
            model.to(CFG.device)
            params = [
                {"params": model.module.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
                {"params": model.module.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
                {"params": model.module.motion_encoder.parameters(), "lr": CFG.motion_encoder_lr},
                {"params": itertools.chain(
                    model.module.image_projection.parameters(), model.module.text_projection.parameters(),
                    model.module.motion_projection.parameters()
                ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
            ]
        else:
            model.to(CFG.device)
            params = [
                {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
                {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
                {"params": model.motion_encoder.parameters(), "lr": CFG.motion_encoder_lr},
                {"params": itertools.chain(
                    model.image_projection.parameters(), model.text_projection.parameters(),
                    model.motion_projection.parameters()
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
            print('loss:', train_loss)
            if train_loss.avg < best_loss:
                best_loss = train_loss.avg
                torch.save(model.state_dict(), "final.pt")
                print("Finish")

            lr_scheduler.step(train_loss.avg)


if __name__ == '__main__':
    main()
