# %%
from torch import nn

# %%
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(208, 208 * 8),
            nn.ReLU(),
            nn.Linear(208 * 8, 208 * 8),
            nn.ReLU(),
            nn.Linear(208 * 8, 208),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x_sum = x.sum(dim=1).unsqueeze(1)
        x = self.model(x)
        #   Model does a softmax, so sum is 1, here we force the expected sum
        #   (i.e. the number of cards that is left)
        x = x * (x_sum - 1)
        return x
# %%
