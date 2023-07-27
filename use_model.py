# %%
import torch as t

# %%
from models import SimpleModel
from utils import pbn_to_repr, repr_to_pbn

model = SimpleModel(2)
model.load_state_dict(t.load("model_2_500_20.pth"))
# %%
pbn = '2.2.. KQ... JT... 98...'
in_ = pbn_to_repr(pbn)
out = model(in_.unsqueeze(0))[0].round()
out_pbn = repr_to_pbn(out)
print(pbn)
print(out_pbn)
# %%
