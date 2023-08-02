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
        5: [299],
    },
)

# %%
model = TransformerModel(
    d_model=256,
    nhead=8,
    num_layers=4,
).to(device)

batch_size = 1024
epochs = 100

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=1000, shuffle=True)

optimizer = t.optim.Adam(model.parameters(), lr=0.0001)
train_loss_list = []
train_accuracy = []
test_loss_list = []
test_accuracy = []

def get_loss_and_acc(model, inputs, labels):
    inputs = inputs.to(device)
    labels = labels[:, :52].to(device)
    
    preds = model(inputs)
        
    loss = F.mse_loss(preds, labels)
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
t.save(model.state_dict(), "transformer_2345c_wt.pth")

# %%
model.eval()
for test_inputs, test_labels in iter(test_loader):
    test_loss, test_acc = get_loss_and_acc(model, test_inputs, test_labels)
    print(test_acc)
# %%
