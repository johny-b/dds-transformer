# %%
from torch import nn

# %%
class SimpleModel(nn.Module):
    def __init__(self, num_cards: int):
        super().__init__()
        self.num_cards = num_cards
        self.model = nn.Sequential(
            nn.Linear(208, 208 * 8),
            nn.ReLU(),
            nn.Linear(208 * 8, 208 * 8),
            nn.ReLU(),
            nn.Linear(208 * 8, 208),
            nn.Softmax(dim=1),
        )
        
    def forward(self, x):
        x = self.model(x)
        #   Model does a softmax, so sum is 1, here we force the expected sum
        #   (i.e. the number of cards that is left)
        x = x * (self.num_cards * 4 - 1)
        return x
