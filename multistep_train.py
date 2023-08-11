# %%
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import SimpleModel, TransformerModel
from datasets import Dataset

device = "cuda"
writer = SummaryWriter('runs')

# %%

use_amp = True
batch_size = 1024 * 2
num_suits = 1
epochs = 100

# %%
def get_data(card_cnt, dataset_size):
    trainset = Dataset(
        {
            card_cnt: list(range(dataset_size)),
        },
        card_tricks=True,
    )
    testset = Dataset(
        {
            card_cnt: [400, 401, 402],
        },
        card_tricks=True,
    )
    test_data = list(iter(DataLoader(testset, batch_size=batch_size)))
    return trainset, test_data


def get_loss_and_acc(model, inputs, labels):
    inputs = inputs.to(device)
    labels = labels[:,:num_suits * 52].to(device)
    mask = labels >= 0

    with t.autocast(device_type='cuda', dtype=t.float16, enabled=use_amp):
        preds = model(inputs)[:,:num_suits * 52]
        preds = preds * mask
        labels = labels * mask
        loss = F.mse_loss(preds, labels)

    round_preds = preds.round()

    # acc = (round_preds == labels).all(dim=1).mean(dtype=t.float32).item()
    acc_2 = (((round_preds == labels) * mask).sum() / mask.sum()).round(decimals=3).item()

    return loss, acc_2


def train(model, trainset, test_data, total_batch_ix, max_batches):
    start_batch_ix = total_batch_ix
    
    optimizer = t.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
    scaler = t.cuda.amp.GradScaler(enabled=use_amp)
    
    train_loader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=12, 
        pin_memory=True, 
        drop_last=True,
        persistent_workers=True,
    
    )
    while True:
        for inputs, labels, _ in tqdm(train_loader):
            model.train()

            loss, acc = get_loss_and_acc(model, inputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()

            total_batch_ix += 1
    
            writer.add_scalars(
                'batch',
                {
                    'loss': loss.item(),
                    'acc': acc,
                },
                total_batch_ix,
            )

            if not (total_batch_ix % 5000):
                del inputs
                del labels
                model.eval()

                loss_list = []
                acc_list = []

                for test_inputs, test_labels, num_tricks in test_data:
                    test_loss, test_acc = get_loss_and_acc(model, test_inputs, test_labels)
                    loss_list.append(test_loss.item())
                    acc_list.append(test_acc)
                    

                test_loss = sum(loss_list)/len(loss_list)
                test_acc = sum(acc_list)/len(acc_list)
                
                writer.add_scalars(
                    'loss', 
                    {'train': loss.item(), 'test': test_loss},
                    total_batch_ix,
                )
                writer.add_scalars(
                    'accuracy',
                    {'acc': test_acc},
                    total_batch_ix,
                )
                
                if total_batch_ix - start_batch_ix >= max_batches:
                    return test_acc, test_loss, total_batch_ix
                

# %%
prev_model = None
num_batches = 0
for card_cnt in range(2, 14):
    trainset, test_data = get_data(card_cnt, 100)
    model = TransformerModel(
        d_model=512,
        nhead=32,
        num_layers=3,
        end_model=prev_model,
    ).to(device)
    accuracy, loss, num_batches = train(
        model, 
        trainset, 
        test_data, 
        num_batches, 
        max_batches=10000,
    )
    
    t.save(model.state_dict(), f"m_{card_cnt}_{accuracy}_{loss}.pth")
    
    prev_model = model
    prev_model.requires_grad_(False)
# %%
