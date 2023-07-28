# %%
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

from tqdm import tqdm

import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import SimpleModel
from datasets import SingleSizeDataset

device = "cuda"


# %%
NUM_CARDS = 2

trainset = SingleSizeDataset(NUM_CARDS, list(range(5)))
testset = SingleSizeDataset(NUM_CARDS, [299])

# %%
model = SimpleModel().to(device)
	
#   higher batch size (4096) -> slower learning
batch_size = 256

#   300 epochs on 100 000 training got us ~ 97.5% test accuracy, but was still improving
epochs = 10

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=8192)

optimizer = t.optim.AdamW(model.parameters(), weight_decay=0.1)
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
    
	
for epoch in tqdm(range(epochs)):
    for inputs, labels in train_loader:
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

t.save(model.state_dict(), f"model_{NUM_CARDS}_{epochs}_{len(trainset.file_ids)}.pth")
# %%
for ix, acc in enumerate(test_accuracy):
    print(ix, acc)

# %%
