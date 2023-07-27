# %%
from tqdm import tqdm

import torch as t
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

device = "cuda"


# %%
class Dataset2card(Dataset):
    def __init__(self, file_ids: list[int]):
        self.file_ids = file_ids
        self.pbn_data = self._get_pbn_data()
        
        self.data = []
        for in_pbn, out_pbn in tqdm(self.pbn_data):
            self.data.append((self._parse_pbn(in_pbn), self._parse_pbn(out_pbn)))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, ix):
        return self.data[ix]
    
    def _get_pbn_data(self):
        data = []
        for file_ix in tqdm(self.file_ids):
            fname = f"data/boards_2_card/{file_ix}.csv"
            data += self._get_file_data(fname)
        return data
    
    def _get_file_data(self, fname):
        data = []
        with open(fname, 'r') as f:
            rows = [row.strip() for row in f.readlines()]
            data += [self._parse_row(row) for row in rows]
        return data
    
    def _parse_row(self, row):
        parts = row.split(';')
        pbn, card = parts[2], parts[3]
        
        out_pbn = self._remove_card(pbn, card)
        
        return pbn, out_pbn
    
    def _remove_card(self, pbn, card):
        def this_card(suit: int, val: str, card: str) -> bool:
            suits = "♠♥♦♣"
            this_card = suits[suit] + val
            return this_card == card

        pbn = pbn[2:]
        
        new_hands = []
        for hand in pbn.split(' '):
            suits = []
            for suit_ix, suit_cards in enumerate(hand.split('.')):
                for ix, val in enumerate(suit_cards):
                    if this_card(suit_ix, val, card):
                        new_suit = suit_cards[:ix] + suit_cards[ix + 1:]
                        break
                else:
                    new_suit = suit_cards
                suits.append(new_suit)
            new_hands.append(".".join(suits))
        
        new_pbn = " ".join(new_hands)
        assert len(new_pbn) + 1 == len(pbn), f"{pbn} + {card} -> {new_pbn}" 
        return "N:" + new_pbn
    
    def _parse_pbn(self, pbn):
        #   Return bool tensor of length 204
        hands = pbn[2:].split(' ')
        return t.cat([self._parse_pbn_hand(hand) for hand in hands])
    
    def _parse_pbn_hand(self, hand):
        suits = hand.split('.')
        return t.cat([self._parse_pbn_suit(suit) for suit in suits])
    
    def _parse_pbn_suit(self, suit):
        cards = list(str(x) for x in range(2, 10)) + list('TJQKA')
        cards.reverse()
        has_card = [card in suit for card in cards]
        return t.tensor(has_card, dtype=t.float32)

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
        x = self.model(x)
        #   model does a softmax, so sum is 1, and we know sum should be 7
        x = x * 7
        return x

# %%

trainset = Dataset2card([0,1,2,3,4,5,6,7,8,9])
testset = Dataset2card([81])

# %%
model = SimpleModel().to(device)
	
#   higher batch size (4096) -> slower learning
batch_size = 1024

#   this got us ~ 97.5% test accuracy, but was still improving
epochs = 300

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size)


optimizer = t.optim.Adam(model.parameters())
train_loss_list = []
train_accuracy = []
test_loss_list = []
test_accuracy = []
	
for epoch in tqdm(range(epochs)):
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        pred = model(inputs)
        loss = F.mse_loss(pred, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss_list.append(loss.item())
        
        round_pred = pred.round()
        acc = (round_pred == labels).all(dim=1).mean(dtype=t.float32)
        train_accuracy.append(acc.item())
        
    test_inputs, test_labels = next(iter(test_loader))
    test_inputs = test_inputs.to(device)
    test_labels = test_labels.to(device)
    test_pred = model(test_inputs)
    
    test_loss = F.mse_loss(test_pred, test_labels)
    test_loss_list.append(test_loss.item())
    
    round_test_pred = test_pred.round()
    test_acc = (round_test_pred == test_labels).all(dim=1).mean(dtype=t.float32)
    test_accuracy.append(test_acc.item())
    
    

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
plt.plot(list(range(len(test_loss_list[250:]))), test_loss_list[250:])
plt.show()

# %%
