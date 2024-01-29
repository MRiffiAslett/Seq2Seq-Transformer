import click
from pathlib import Path
from einops import rearrange
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from torch.optim import Adam
from Optim import ScheduledOptim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ParallelLanguageDataset
from model import LanguageTransformer

# Command-line interface for setting training parameters
@click.command()
@click.argument('epochs', type=int, default=20)
@click.argument('max_seq_len', type=int, default=96)
@click.argument('tokens_per_batch', type=int, default=2000)
@click.argument('vocab_size', type=int, default=10000 + 4)
@click.argument('model_dim', type=int, default=512)
@click.argument('encoder_layers', type=int, default=6)
@click.argument('decoder_layers', type=int, default=6)
@click.argument('feedforward_dim', type=int, default=2048)
@click.argument('num_heads', type=int, default=8)
@click.argument('pos_dropout_rate', type=float, default=0.1)
@click.argument('trans_dropout_rate', type=float, default=0.1)
@click.argument('warmup_steps', type=int, default=4000)
def main(**kwargs):
    project_path = str(Path(__file__).resolve().parents[0])

    # Create train and validation datasets
    train_dataset = ParallelLanguageDataset(project_path + '/data/processed/en/train.pkl',
                                            project_path + '/data/processed/fr/train.pkl',
                                            kwargs['tokens_per_batch'], kwargs['max_seq_len'])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    valid_dataset = ParallelLanguageDataset(project_path + '/data/processed/en/val.pkl',
                                            project_path + '/data/processed/fr/val.pkl',
                                            kwargs['tokens_per_batch'], kwargs['max_seq_len'])
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize the transformer model
    model = LanguageTransformer(kwargs['vocab_size'], kwargs['model_dim'], kwargs['num_heads'], kwargs['encoder_layers'],
                                kwargs['decoder_layers'], kwargs['feedforward_dim'], kwargs['max_seq_len'],
                                kwargs['pos_dropout_rate'], kwargs['trans_dropout_rate']).to('cuda')

    # Use Xavier normal initialization in the transformer
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_normal_(param)

    # Initialize the optimizer with scheduled learning rate
    optim = ScheduledOptim(
        Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        kwargs['model_dim'], kwargs['warmup_steps'])

    # Use cross-entropy loss, ignoring any padding
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Train the model and store losses
    train_losses, val_losses = train(train_loader, valid_loader, model, optim, criterion, kwargs['epochs'])

# Function for training the transformer model
def train(train_loader, valid_loader, model, optim, criterion, num_epochs):
    print_every = 500
    model.train()

    lowest_val = 1e9
    train_losses = []
    val_losses = []
    total_step = 0

    for epoch in range(num_epochs):
        pbar = tqdm(total=print_every, leave=False)
        total_loss = 0

        # Shuffle batches every epoch
        train_loader.dataset.shuffle_batches()

        for step, (src, src_key_padding_mask, tgt, tgt_key_padding_mask) in enumerate(iter(train_loader)):
            total_step += 1

            # Send the batches and key_padding_masks to GPU
            src, src_key_padding_mask = src[0].to('cuda'), src_key_padding_mask[0].to('cuda')
            tgt, tgt_key_padding_mask = tgt[0].to('cuda'), tgt_key_padding_mask[0].to('cuda')
            memory_key_padding_mask = src_key_padding_mask.clone()

            # Create tgt_inp and tgt_out (which is tgt_inp but shifted by 1)
            tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:]
            tgt_mask = gen_nopeek_mask(tgt_inp.shape[1]).to('cuda')

            # Forward pass
            optim.zero_grad()
            outputs = model(src, tgt_inp, src_key_padding_mask, tgt_key_padding_mask[:, :-1], memory_key_padding_mask, tgt_mask)
            loss = criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_out, 'b o -> (b o)'))

            # Backpropagate and update optimizer
            loss.backward()
            optim.step_and_update_lr()

            total_loss += loss.item()
            train_losses.append((step, loss.item()))
            pbar.update(1)

            if step % print_every == print_every - 1:
                pbar.close()
                print(f'Epoch [{epoch + 1} / {num_epochs}] \t Step [{step + 1} / {len(train_loader)}] \t '
                      f'Train Loss: {total_loss / print_every}')
                total_loss = 0
                pbar = tqdm(total=print_every, leave=False)

        # Validate every epoch
        pbar.close()
        val_loss = validate(valid_loader, model, criterion)
        val_losses.append((total_step, val_loss))

        if val_loss < lowest_val:
            lowest_val = val_loss
            torch.save(model, 'output/transformer.pth')

        print(f'Val Loss: {val_loss}')

    return train_losses, val_losses

# Function for validation
def validate(valid_loader, model, criterion):
    pbar = tqdm(total=len(iter(valid_loader)), leave=False)
    model.eval()

    total_loss = 0

    for src, src_key_padding_mask, tgt, tgt_key_padding_mask in iter(valid_loader):
        with torch.no_grad():
            src, src_key_padding_mask = src[0].to('cuda'), src_key_padding_mask[0].to('cuda')
            tgt, tgt_key_padding_mask = tgt[0].to('cuda'), tgt_key_padding_mask[0].to('cuda')
            memory_key_padding_mask = src_key_padding_mask.clone()
            tgt_inp = tgt[:, :-1]
            tgt_out = tgt[:, 1:].contiguous()
            tgt_mask = gen_nopeek_mask(tgt_inp.shape[1]).to('cuda')

            outputs = model(src, tgt_inp, src_key_padding_mask, tgt_key_padding_mask[:, :-1], memory_key_padding_mask, tgt_mask)
            loss = criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_out, 'b o -> (b o)'))

            total_loss += loss.item()
            pbar.update(1)

    pbar.close()
    model.train()
    return total_loss / len(valid_loader)

# Function to generate the nopeek mask
def gen_nopeek_mask(length):
    mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask

if __name__ == "__main__":
    main()
