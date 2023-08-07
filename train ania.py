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

from models import TransformerModel, TrickModel
from datasets import Dataset

device = "cuda"
writer = SummaryWriter('runs')

# %%
import neptune

run = neptune.init_run(
    project='aniasztyber/dds-transformer',
    tags=['tricks', 'backbone'],
    source_files=["train ania.py", "models.py"])

# %%
params = {
    'n_cards': 13,
    'd_model': 256,
    'nhead': 8,
    'num_layers': 12,
    'lr': 0.0001,
    'weight_decay': 0.1,
    'batch_size': 1024,
    'epochs': 10
}

run['params'] = params

# %%

trainset = Dataset(
    {
        params['n_cards']: list(range(100)),
    },
)

testset = Dataset(
    {
        params['n_cards']: [297],
    },
)

# %%
base_model = TransformerModel(
    d_model=params['d_model'],
    nhead=params['nhead'],
    num_layers=params['num_layers'],
)
base_model.load_state_dict(t.load("transformer_13c_12l_859.pth"))
base_model.eval()
base_model = base_model.to(device)

model = TrickModel(base_model)
model = model.to(device)

# %%

use_amp = True
optimizer = t.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
scaler = t.cuda.amp.GradScaler(enabled=use_amp)

# %%
# for g in optimizer.param_groups:
#     g['lr'] = 0.00005

# model.load_state_dict(t.load("transformer_5c_wt_all_12_layers_798.pth"))
batch_size = params['batch_size']
epochs = params['epochs']

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=3)

def get_loss_and_acc(model, inputs, labels, tricks_labels):
    inputs = inputs.to(device)
    labels = labels[:, :52].to(device)
    tricks_labels = tricks_labels.to(t.float32).unsqueeze(1).to(device)
    
    with t.autocast(device_type='cuda', dtype=t.float16, enabled=use_amp):    
        # preds, tricks_preds = model(inputs)
        tricks_preds = model(inputs)
        # card_loss = F.binary_cross_entropy_with_logits(preds, labels)
        tricks_loss = F.mse_loss(tricks_preds, tricks_labels)

    round_preds = tricks_preds.round()
    acc = (round_preds == tricks_labels).all(dim=1).mean(dtype=t.float32).item()
    
    return tricks_loss, acc
    
total_batch_ix = 0
for epoch_ix, epoch in enumerate(tqdm(range(epochs))):
    model.train()
    
    for batch_ix, (inputs, labels, tricks_labels) in enumerate(train_loader):
        tricks_loss, acc = get_loss_and_acc(model, inputs, labels, tricks_labels)
        scaler.scale(tricks_loss).backward()
        scaler.step(optimizer)
        optimizer.zero_grad()
        scaler.update()
        
        writer.add_scalars(
            'batch',
            {
                # 'card_loss': card_loss.item(),
                'trick_loss': tricks_loss.item(),
                'accuracy': acc,
            },
            total_batch_ix,
        )
        total_batch_ix += 1

    del inputs
    del labels
    del tricks_labels
    model.eval()

    epoch_loss_list = []
    epoch_acc_list = []
    epoch_tricks_loss = []
    for test_inputs, test_labels, test_tricks_labels in iter(test_loader):
        test_tricks_loss, test_acc = get_loss_and_acc(model, test_inputs, test_labels, test_tricks_labels)
        # epoch_loss_list.append(test_loss.item())
        epoch_acc_list.append(test_acc)
        epoch_tricks_loss.append(test_tricks_loss.item())
    
    '''
    writer.add_scalars(
        'card_loss', 
        {'train': card_loss.item(), 
         'test': sum(epoch_loss_list)/len(epoch_loss_list)}, 
        epoch_ix,
    )
    '''
    writer.add_scalars(
        'trick_loss', 
        {'train': tricks_loss.item(), 
         'test': sum(epoch_tricks_loss)/len(epoch_tricks_loss)}, 
        epoch_ix,
    )
    
    writer.add_scalars(
        'accuracy', 
        {'train': acc, 
         'test': sum(epoch_acc_list)/len(epoch_acc_list)}, 
        epoch_ix,
    )
    
# %%
del test_inputs
del test_labels
    
# %%
file_name = "transformer_13c_bacbone_tricks.pth"
t.save(model.state_dict(), file_name)
run["model"].upload(file_name)
# %%
model.eval()
accs = []
trick_losses = []
for test_inputs, test_labels, test_tricks_labels in iter(test_loader):
    test_tricks_loss, test_acc = get_loss_and_acc(model, test_inputs, test_labels, test_tricks_labels)
    accs.append(test_acc)
    trick_losses.append(test_tricks_loss.item())
    
avg_acc = sum(accs)/len(accs)
avg_trick_loss = sum(trick_losses)/len(trick_losses)
print(avg_acc)
print(avg_trick_loss)
run['eval/card_optim_acc'] = avg_acc
run['eval/trick_mse_loss'] = avg_trick_loss
# %%
run.stop()

# %%
