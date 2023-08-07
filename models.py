# %%
import torch as t
from torch import nn

# %%
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(208, 208 * 8),
            nn.ReLU(),
            nn.Linear(208 * 8, 208 * 8),
            nn.Dropout(p=0.9),
            nn.ReLU(),
            nn.Linear(208 * 8, 208),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return x - self.model(x)
        
# %%
class TransformerModel(nn.Module):
    def __init__(self, *, d_model, nhead, num_layers):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        self.embed = nn.Linear(52, d_model)
        self.pos_embed = nn.Parameter(t.empty(4 * d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        self.trick_embed = nn.Linear(52, d_model)
        self.trick_pos_embed = nn.Parameter(t.empty(3 * d_model))
        nn.init.normal_(self.trick_pos_embed, std=0.02)

        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers,
        )
        self.unembed = nn.Linear(7 * d_model, 52)
        # self.final_act = nn.Sigmoid()
        
    def encode(self, x: t.Tensor):
        assert len(x.shape) == 2
        assert x.shape[1] == 364
        in_hand = x[:,:52]
        
        board = x[:,:208].reshape((x.shape[0], 4, 52))
        trick = x[:,208:].reshape((x.shape[0], 3, 52))

        board = t.stack((
            self.embed(board[:,0,:]),
            self.embed(board[:,1,:]),
            self.embed(board[:,2,:]),
            self.embed(board[:,3,:]),
        )).permute((1,0,2))
        
        pos_embed = self.pos_embed.reshape((4, self.d_model)).unsqueeze(0)
        board = board + pos_embed
        
        trick = t.stack((
            self.trick_embed(trick[:,0,:]),
            self.trick_embed(trick[:,1,:]),
            self.trick_embed(trick[:,2,:]),
        )).permute((1,0,2))
        trick_pos_embed = self.trick_pos_embed.reshape((3, self.d_model)).unsqueeze(0)
        trick = trick + trick_pos_embed
        
        x = t.cat([board, trick], dim=1)
        x = self.enc(x)
        return x
    
    def forward(self, x: t.Tensor):
        x = self.encode(x)
        x = x.flatten(start_dim=1)
        x = self.unembed(x)
        # x = self.final_act(x)
        return x

# %%
class TransformerModelWithTricks(nn.Module):
    def __init__(self, *, d_model, nhead, num_layers):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        self.embed = nn.Linear(52, d_model)
        self.pos_embed = nn.Parameter(t.empty(4 * d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        self.trick_embed = nn.Linear(52, d_model)
        self.trick_pos_embed = nn.Parameter(t.empty(3 * d_model))
        nn.init.normal_(self.trick_pos_embed, std=0.02)

        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers,
        )
        self.unembed = nn.Linear(7 * d_model, 52)
        
        self.unembed_tricks = nn.Linear(7 * d_model, 1)
        # self.final_act = nn.Sigmoid()
    
    def forward(self, x: t.Tensor):
        assert len(x.shape) == 2
        assert x.shape[1] == 364
        in_hand = x[:,:52]
        
        board = x[:,:208].reshape((x.shape[0], 4, 52))
        trick = x[:,208:].reshape((x.shape[0], 3, 52))

        board = t.stack((
            self.embed(board[:,0,:]),
            self.embed(board[:,1,:]),
            self.embed(board[:,2,:]),
            self.embed(board[:,3,:]),
        )).permute((1,0,2))
        
        pos_embed = self.pos_embed.reshape((4, self.d_model)).unsqueeze(0)
        board = board + pos_embed
        
        trick = t.stack((
            self.trick_embed(trick[:,0,:]),
            self.trick_embed(trick[:,1,:]),
            self.trick_embed(trick[:,2,:]),
        )).permute((1,0,2))
        trick_pos_embed = self.trick_pos_embed.reshape((3, self.d_model)).unsqueeze(0)
        trick = trick + trick_pos_embed
        
        x = t.cat([board, trick], dim=1)
        x = self.enc(x)
        x = x.flatten(start_dim=1)
        out = self.unembed(x)
        # x = self.final_act(x)
        
        tricks = self.unembed_tricks(x)

        return out, tricks
# %%

class TrickModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base_model = model
        self.unembed_tricks = nn.Linear(7 * self.base_model.d_model, 1)
        
    def forward(self, x):
        x = self.base_model.encode(x)
        x = x.flatten(start_dim=1)
        x = self.unembed_tricks(x)
        return x

# %%
