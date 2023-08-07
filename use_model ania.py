# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch as t
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import TransformerModelWithTricks
from datasets import Dataset


# %%
device = "cpu"

# %%
model = TransformerModelWithTricks(
    d_model=256,
    nhead=8,
    num_layers=4,
)
model.load_state_dict(t.load("transformer_5c_wt_all_tricks.pth", map_location=t.device('cpu')))
model.eval()
model = model.to(device)

# %%
testset = Dataset(
    {
        5: [297, 298, 299],
    },
)

# %%
batch_size = 1000
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)

accs = []
for test_inputs, test_labels, test_tricks_labels in iter(test_loader):
    test_inputs = test_inputs.to(device)
    test_tricks_labels = test_tricks_labels.unsqueeze(1).to(device)
    preds, tricks_preds = model(test_inputs)
    acc = (tricks_preds.round() == test_tricks_labels).to(t.float32).mean()
    accs.append(acc.item())
# %%
print(sum(accs)/len(accs))

# %%
