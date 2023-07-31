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
        5: list(range(99)),
    },
)
testset = Dataset(
    {
        5: [99],
    },
)

# %%

trainset_small = Dataset(
    {
        5: list(range(10)),
    },
)
# %%
model = TransformerModel(
    d_model=64,
    nhead=8,
    num_layers=2,
).to(device)

batch_size = 4096
epochs = 100

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=1000, shuffle=True)

optimizer = t.optim.Adam(model.parameters(), lr=0.002)
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
    
	
for epoch_ix, epoch in enumerate(tqdm(range(epochs))):
    model.train()
    
    for i, (inputs, labels) in enumerate(train_loader):
        loss, acc = get_loss_and_acc(model, inputs, labels)
        train_loss_list.append(loss.item())
        train_accuracy.append(acc)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    test_inputs, test_labels = next(iter(test_loader))
    test_loss, test_acc = get_loss_and_acc(model, test_inputs, test_labels)
    test_loss_list.append(test_loss.item())
    test_accuracy.append(test_acc)
    
    writer.add_scalars(
        'loss', 
        {'train': train_loss_list[-1], 
         'test': test_loss_list[-1],}, 
        epoch_ix,
    )
    writer.add_scalars(
        'accuracy', 
        {'train': train_accuracy[-1], 
         'test': test_accuracy[-1],}, 
        epoch_ix,
    )
    
# %%
t.save(model.state_dict(), "transformer.pth")

# %%

from matplotlib import pyplot as plt
plt.plot(list(range(len(train_loss_list))), train_loss_list)
plt.show()

plt.plot(list(range(len(train_accuracy))), train_accuracy)
plt.show()

plt.plot(list(range(len(test_loss_list))), test_loss_list)
plt.show()

plt.plot(list(range(len(test_accuracy))), test_accuracy)
plt.show()

# %%
