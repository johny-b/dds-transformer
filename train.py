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
        5: list(range(20)),
    },
)

testset = Dataset(
    {
        5: [297, 298, 299],
    },
)

# %%
model = TransformerModel(
    d_model=256,
    nhead=8,
    num_layers=12,
).to(device)

# %%

optimizer = t.optim.Adam(model.parameters(), lr=0.0001)
# %%
# for g in optimizer.param_groups:
#     g['lr'] = 0.00005

# model.load_state_dict(t.load("transformer_5c_wt_all_12_layers_798.pth"))
batch_size = 32
epochs = 100

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=1000, shuffle=True)

def get_loss_and_acc(model, inputs, labels):
    inputs = inputs.to(device)
    labels = labels[:, :52].to(device)
    
    preds = model(inputs)
        
    loss = F.binary_cross_entropy(preds, labels)
    round_preds = preds.round()
    acc = (round_preds == labels).all(dim=1).mean(dtype=t.float32).item()
    
    return loss, acc
    
total_batch_ix = 0
for epoch_ix, epoch in enumerate(tqdm(range(epochs))):
    model.train()
    
    for batch_ix, (inputs, labels) in enumerate(train_loader):
        loss, acc = get_loss_and_acc(model, inputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        writer.add_scalars(
            'batch',
            {
                'loss': loss.item(),
                'accuracy': acc,
            },
            total_batch_ix,
        )
        total_batch_ix += 1

    model.eval()

    epoch_loss_list = []
    epoch_acc_list = []
    for test_inputs, test_labels in iter(test_loader):
        test_loss, test_acc = get_loss_and_acc(model, test_inputs, test_labels)
        epoch_loss_list.append(test_loss.item())
        epoch_acc_list.append(test_acc)
    
    writer.add_scalars(
        'loss', 
        {'train': loss.item(), 
         'test': sum(epoch_loss_list)/len(epoch_loss_list)}, 
        epoch_ix,
    )
    writer.add_scalars(
        'accuracy', 
        {'train': acc, 
         'test': sum(epoch_acc_list)/len(epoch_acc_list)}, 
        epoch_ix,
    )
    
# %%
del inputs
del labels
del test_inputs
del test_labels
    
# %%
t.save(model.state_dict(), "transformer_5c_wt_all_12_layers_798.pth")

# %%
model.eval()
for test_inputs, test_labels in iter(test_loader):
    test_loss, test_acc = get_loss_and_acc(model, test_inputs, test_labels)
    print(test_acc)
# %%
