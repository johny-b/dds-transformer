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
        13: list(range(231)) + list(range(1000, 1065)) + list(range(2000, 2099)),
    },
    card_tricks=True,
)

testset = Dataset(
    {
        13: [231, 1065, 2099],
    },
    card_tricks=True,
)

# %%
model = TransformerModel(
    d_model=256,
    nhead=8,
    num_layers=12,
).to(device)

# %%

use_amp = True
optimizer = t.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
scaler = t.cuda.amp.GradScaler(enabled=use_amp)

# %%
# for g in optimizer.param_groups:
#     g['lr'] = 0.001

# model.load_state_dict(t.load("transformer_full_12l_s_epoch_5.pth"))
# optimizer.zero_grad()
# scaler.update()
# %%
batch_size = 2048
epochs = 100

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True, drop_last=True, prefetch_factor=5)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True, drop_last=True, prefetch_factor=5)

def get_loss_and_acc(model, inputs, labels, batch_ix=None):
    inputs = inputs.to(device)
    labels = labels[:, :52].to(device)
    mask = labels >= 0

    with t.autocast(device_type='cuda', dtype=t.float16, enabled=use_amp):
        preds = model(inputs)
        preds = preds * mask
        labels = labels * mask
        loss = F.mse_loss(preds, labels)

    round_preds = preds.round()
    if batch_ix is not None and not batch_ix % 20:
        print([int(x) for x in labels[0]])
        print([int(x) for x in round_preds[0]])
        print()
    acc = (round_preds == labels).all(dim=1).mean(dtype=t.float32).item()

    return loss, acc

total_batch_ix = 0
for epoch_ix, epoch in enumerate(tqdm(range(epochs))):
    for batch_ix, (inputs, labels, num_tricks) in enumerate(train_loader):
        loss, acc = get_loss_and_acc(model, inputs, labels, batch_ix)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        optimizer.zero_grad()
        scaler.update()

        writer.add_scalars(
            'batch',
            {
                'loss': loss.item(),
                'accuracy': acc,
            },
            total_batch_ix,
        )
        total_batch_ix += 1

        if not (total_batch_ix % 5000):
            del inputs
            del labels
            model.eval()

            epoch_loss_list = []
            epoch_acc_list = []
            for test_inputs, test_labels, num_tricks in iter(test_loader):
                test_loss, test_acc = get_loss_and_acc(model, test_inputs, test_labels)
                epoch_loss_list.append(test_loss.item())
                epoch_acc_list.append(test_acc)

            test_loss = sum(epoch_loss_list)/len(epoch_loss_list)
            test_acc = sum(epoch_acc_list)/len(epoch_acc_list)
            writer.add_scalars(
                'loss',
                {'train': loss.item(),
                'test': test_loss},
                total_batch_ix,
            )
            writer.add_scalars(
                'accuracy',
                {'train': acc,
                'test': test_acc},
                total_batch_ix,
            )
            model.train()

        if not total_batch_ix % 50000:
            t.save(model.state_dict(), f"transformer_full_12l_nt_{test_loss}_{test_acc}.pth")

# %%
del loss
del test_loss
del inputs
del labels
del test_inputs
del test_labels

# %%
t.save(model.state_dict(), "model_13c_full_nt_560x2000_782.pth")

# %%
model.eval()
for test_inputs, test_labels in iter(test_loader):
    test_loss, test_acc = get_loss_and_acc(model, test_inputs, test_labels)
    print(test_acc)
# %%
