# %%
import torch as t

# %%
from models import SimpleModel
from utils import pbn_to_repr, repr_to_pbn

model = SimpleModel()
model.load_state_dict(t.load("model_2_300_299.pth"))
# %%
pbn = '2.6.. A.5.. ...34 98...'
in_ = pbn_to_repr(pbn)
out = model(in_.unsqueeze(0))[0].round()
out_pbn = repr_to_pbn(out)
print(pbn)
print(out_pbn)
# %%
