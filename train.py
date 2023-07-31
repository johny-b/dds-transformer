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
        2: list(range(10)),
    },
)
testset = Dataset(
    {
        2: [99],
    },
)

# %%
from torch import nn
  

class TransformerModel(nn.Module):
    def __init__(self, d_model=24, nhead=12):
        super().__init__()
        self.embed = nn.Linear(52, d_model)
        self.enc_1 = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.enc_2 = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.enc_3 = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.enc_4 = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.unembed = nn.Linear(4 * d_model, 208)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: t.Tensor):
        assert len(x.shape) == 2
        assert x.shape[1] == 208
        
        x = x.reshape((x.shape[0], 4, 52))

        num_cards = x.flatten(start_dim=1).sum(dim=1)
        expected_num_cards = (num_cards - 1).unsqueeze(1)
        
        x = t.stack((
            self.embed(x[:,0,:]),
            self.embed(x[:,1,:]),
            self.embed(x[:,2,:]),
            self.embed(x[:,3,:]),
        )).permute((1,0,2))
        
        x = self.enc_1(x)
        x = self.enc_2(x)
        x = self.enc_3(x)
        x = self.enc_4(x)
        x = x.flatten(start_dim=1)
        x = self.unembed(x)
        x = self.softmax(x)
        x = x * expected_num_cards

        return x
        
model = TransformerModel().to(device)

batch_size = 128
epochs = 100

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)

optimizer = t.optim.AdamW(model.parameters(), weight_decay=0.02, lr=0.0002)
train_loss_list = []
train_accuracy = []
test_loss_list = []
test_accuracy = []

def get_loss_and_acc(model, inputs, labels):
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    preds = model(inputs)
        
    loss = F.mse_loss(preds, labels)
    round_preds = preds.round()
    acc = (round_preds == labels).all(dim=1).mean(dtype=t.float32).item()
    
    return loss, acc
    
	
for epoch_ix, epoch in enumerate(tqdm(range(epochs))):
    for i, (inputs, labels) in enumerate(train_loader):
        loss, acc = get_loss_and_acc(model, inputs, labels)
        train_loss_list.append(loss.item())
        train_accuracy.append(acc)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

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
