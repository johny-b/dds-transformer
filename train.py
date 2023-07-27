# %%
from tqdm import tqdm

import torch as t
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

device = "cuda"


# %%
class SingleCardDataset(Dataset):
    def __init__(self, num_cards: int, file_ids: list[int]):
        self.num_cards = num_cards
        self.file_ids = file_ids
        
        print("Reading input files")
        self.pbn_data = self._get_pbn_data()
        
        print("Processing boards")
        self.data = []
        for in_pbn, out_pbn in tqdm(self.pbn_data):
            in_ = self._parse_pbn(in_pbn)
            out = self._parse_pbn(out_pbn)
            assert in_.sum().item() == self.num_cards * 4
            assert out.sum().item() == self.num_cards * 4 - 1
            self.data.append((in_, out))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, ix):
        return self.data[ix]
    
    def _get_pbn_data(self):
        data = []
        for file_ix in tqdm(self.file_ids):
            fname = f"data/boards_{self.num_cards}_card/{file_ix}.csv"
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

# %%

NUM_CARDS = 2

trainset = SingleCardDataset(NUM_CARDS, list(range(20)))
testset = SingleCardDataset(NUM_CARDS, [81])

# %%
model = SimpleModel(NUM_CARDS).to(device)
	
#   higher batch size (4096) -> slower learning
batch_size = 1024

#   300 epochs got us ~ 97.5% test accuracy, but was still improving
epochs = 500

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size)

optimizer = t.optim.Adam(model.parameters())
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
