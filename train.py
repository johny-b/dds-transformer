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

trainset = Dataset(
    {
        13: list(range(1000)),
    },
    card_tricks=True,
)

testset = Dataset(
    {
        13: [1001, 1002, 1003],
    },
    card_tricks=True,
)

# %%
model = TransformerModel(
    d_model=256,
    nhead=32,
    num_layers=12,
).to(device)

# %%

use_amp = True
optimizer = t.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)

# %%
scaler = t.cuda.amp.GradScaler(enabled=use_amp)
# %%
# model.load_state_dict(t.load("ttt_2.pth"))
# optimizer.load_state_dict(t.load("opt.pth"))

# %%
# for g in optimizer.param_groups:
#   g['lr'] = 0.0004

# %%
optimizer.zero_grad()
# scaler.update()
# %%
batch_size = 1024 * 2
epochs = 100

train_loader = DataLoader(
    trainset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=12, 
    pin_memory=True, 
    drop_last=True,
    persistent_workers=True,
)
test_data = list(iter(DataLoader(testset, batch_size=batch_size)))
print("Test data ready")

NUM_SUITS = 5  # 1: NT, 2: NT + spades, ...

def get_loss_and_acc(model, inputs, labels, batch_ix=None, split=False):
    inputs = inputs.to(device)
    labels = labels[:,:NUM_SUITS * 52].to(device)
    mask = labels >= 0

    with t.autocast(device_type='cuda', dtype=t.float16, enabled=use_amp):
        preds = model(inputs)[:,:NUM_SUITS * 52]
        preds = preds * mask
        labels = labels * mask
        loss = F.mse_loss(preds, labels)

    round_preds = preds.round()
    if batch_ix is not None and not batch_ix % 100:
        print([int(x) for x in labels[0]])
        print([int(x) for x in round_preds[0]])
        print()
    
    acc = (round_preds == labels).all(dim=1).mean(dtype=t.float32).item()
    acc_2 = (((round_preds == labels) * mask).sum() / mask.sum()).round(decimals=3).item()

    out = [loss, acc, acc_2]
    
    if split:
        nt_preds, suit_preds = preds[:,:52], preds[:,52:]
        nt_labels, suit_labels = labels[:,:52], labels[:,52:]
        with t.autocast(device_type='cuda', dtype=t.float16, enabled=use_amp):
            nt_loss = F.mse_loss(nt_preds, nt_labels).item()
            suit_loss = F.mse_loss(suit_preds, suit_labels).item()
        out += [nt_loss, suit_loss]
    
    return out

total_batch_ix = 0
for epoch_ix, epoch in enumerate(tqdm(range(epochs))):
    for batch_ix, (inputs, labels, num_tricks) in enumerate(train_loader):
        loss, acc, acc_2 = get_loss_and_acc(model, inputs, labels, total_batch_ix)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        optimizer.zero_grad()
        scaler.update()

        writer.add_scalars(
            'batch',
            {
                'loss': loss.item(),
                'acc': acc,
                'acc_2': acc_2,
            },
            total_batch_ix,
        )
        total_batch_ix += 1

        if not (total_batch_ix % 5000):
            del labels
            model.eval()

            epoch_loss_list = []
            epoch_acc_list = []
            epoch_acc_2_list = []
            nt_loss_list = []
            suit_loss_list = []
            for test_inputs, test_labels, num_tricks in test_data:
                test_loss, test_acc, test_acc_2, loss_nt, loss_suit = get_loss_and_acc(model, test_inputs, test_labels, split=True)
                epoch_loss_list.append(test_loss.item())
                epoch_acc_list.append(test_acc)
                epoch_acc_2_list.append(test_acc_2)
                nt_loss_list.append(loss_nt)
                suit_loss_list.append(loss_suit)

            test_loss = sum(epoch_loss_list)/len(epoch_loss_list)
            test_acc = sum(epoch_acc_list)/len(epoch_acc_list)
            test_acc_2 = sum(epoch_acc_2_list)/len(epoch_acc_2_list)
            
            nt_loss = sum(nt_loss_list) / len(nt_loss_list)
            suit_loss = sum(suit_loss_list) / len(suit_loss_list)
            
            writer.add_scalars(
                'loss',
                {'train': loss.item(),
                'test': test_loss,
                'test_nt': nt_loss,
                'test_suit': suit_loss},
                total_batch_ix,
            )
            writer.add_scalars(
                'accuracy',
                {
                    'acc': test_acc,
                    'acc_2': test_acc_2
                },
                total_batch_ix,
            )
            model.train()

        if not total_batch_ix % 50000:
            t.save(model.state_dict(), f"transformer_12l_256_32_all_suits_{test_loss}_{test_acc}.pth")

 # %%
del loss
del test_loss
del inputs
del labels
del test_inputs
del test_labels
# %%
t.save(model.state_dict(), f"transformer_full_12l_512_nt_{test_loss}_{test_acc}.pth")

# %%
t.save(model.state_dict(), "model.pth")
t.save(optimizer.state_dict(), "opt.pth")

# %%
model.eval()
for test_inputs, test_labels in iter(test_loader):
    test_loss, test_acc = get_loss_and_acc(model, test_inputs, test_labels)
    print(test_acc)
# %%
