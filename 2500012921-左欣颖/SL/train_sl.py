# SL training with CNNModel (compatible with preprocessed data format)
from dataset import MahjongGBDataset
from model import CNNModel
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import os
import time
import sys

def main():
    logdir = 'model/'
    os.makedirs(logdir + 'checkpoint', exist_ok=True)

    # Config
    splitRatio = 0.9
    batchSize = 1024
    lr = 5e-4
    epochs = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}', flush=True)

    # Load dataset
    print('Loading dataset...', flush=True)
    trainDataset = MahjongGBDataset(0, splitRatio, True)
    validateDataset = MahjongGBDataset(splitRatio, 1, False)
    # num_workers=0 to avoid Windows multiprocessing issues
    loader = DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True, num_workers=0)
    vloader = DataLoader(dataset=validateDataset, batch_size=batchSize, shuffle=False, num_workers=0)
    print(f'Train: {len(trainDataset)}, Val: {len(validateDataset)}', flush=True)

    # Model
    model = CNNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training
    for e in range(epochs):
        t0 = time.time()
        total_loss = 0
        n_batches = 0
        for i, d in enumerate(loader):
            obs = d[0].float().to(device)
            mask = d[1].float().to(device)
            act = d[2].long().to(device)
            input_dict = {'is_training': True,
                          'obs': {'observation': obs, 'action_mask': mask}}
            logits = model(input_dict)
            loss = F.cross_entropy(logits, act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            if i % 256 == 0:
                print(f'  Batch {i}/{len(loader)}, loss: {loss.item():.4f}', flush=True)

        # Validation
        correct = 0
        total = 0
        for d in vloader:
            obs = d[0].float().to(device)
            mask = d[1].float().to(device)
            act = d[2].long().to(device)
            input_dict = {'is_training': False,
                          'obs': {'observation': obs, 'action_mask': mask}}
            with torch.no_grad():
                logits = model(input_dict)
                pred = logits.argmax(dim=1)
                correct += (pred == act).sum().item()
                total += act.size(0)
        acc = correct / total
        elapsed = time.time() - t0
        print(f'Epoch {e+1}/{epochs} | Loss: {total_loss/n_batches:.4f} | Val Acc: {acc:.4f} | Time: {elapsed:.1f}s', flush=True)

        # Save checkpoint
        ckpt_path = logdir + f'checkpoint/model_{e+1}.pt'
        torch.save(model.state_dict(), ckpt_path)
        print(f'  Saved: {ckpt_path}', flush=True)

    print('Training complete.', flush=True)

if __name__ == '__main__':
    main()
