# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch as t
from torch.utils.data import DataLoader
from endplay.types import Deal, Card, Denom, Rank, Player
from tqdm import tqdm

from models import SimpleModel, TransformerModel
from datasets import Dataset
from utils import pbn_to_repr, repr_to_pbn, hand_repr_to_pbn

suit_cards = list(str(x) for x in range(2, 10)) + list('TJQKA')
suit_cards.reverse()
CARDS = []
for suit in "♠♥♦♣":
    CARDS += [suit + card for card in suit_cards]

# %%
device = "cuda"

# %%
model = TransformerModel(
    d_model=256,
    nhead=8,
    num_layers=12,
)
model.load_state_dict(t.load("transformer_5c_wt_all_12_layers_993.pth", map_location=t.device('cpu')))
model.eval()
model = model.to(device)

# %%
testset = Dataset(
    {
        5: [1000, 1001, 1002],
    },
)

# %%


# %%
not_solved_cnt = 0
bad_best_card_cnt = 0

for ix in tqdm(range(120000)):
    input, label = testset[ix]
    input, label = input.to(device), label.to(device)
    out = model(input.unsqueeze(0))[0]
    pred = t.nn.functional.sigmoid(out).round()
    if (pred == label).all():
        pass
    else:
        not_solved_cnt += 1 
        # bad_card_ix = (label - pred).argmax()
        # bad_card = CARDS[bad_card_ix]
        # ok_card, tricks = testset.pbn_data[ix][2]
        deal = Deal(repr_to_pbn(input[:208]))
        trick_cards = []
        for x in range(3):
            card_repr = input[208 + 52 * x: 208 + 52 * (x + 1)]
            if card_repr.sum().item():
                card = CARDS[card_repr.argmax()]
                trick_cards.insert(0, card)
        
        if trick_cards:
            #   Add cards to the deal and play them
            first_hand = 4 - len(trick_cards)
            for hand, card in zip(range(first_hand, 5), trick_cards):
                deal[hand].extend([card])        
            deal.first = [Player.north, Player.west, Player.south, Player.east][len(trick_cards)]
            for card in trick_cards:
                deal.play(card)

        if out.argmax() not in (label == 1).nonzero().flatten():
            print("!!!!!!!!!!!!!!!!!!!!")
            bad_best_card_cnt += 1            
            deal.pprint()
            print("CORRECT  ", hand_repr_to_pbn(label))
            print("PREDICTED", hand_repr_to_pbn(pred))
            print("BEST", CARDS[out.argmax()])
            print(out.round(decimals=2))
        
            print()
print("BAD CNT", not_solved_cnt)
print("BAD BEST CARD", bad_best_card_cnt)

# %%
pbn = '2.6.. A.5.. ...34 98...'
in_ = pbn_to_repr(pbn)
out = model(in_.unsqueeze(0))[0].round()
out_pbn = repr_to_pbn(out)
print(pbn)
print(out_pbn)
# %%

def get_acc(model, inputs, labels):
    labels = labels[:, :52]
    
    preds = model(inputs)
    round_preds = preds.round()
    acc = (round_preds == labels).all(dim=1).mean(dtype=t.float32).item()
    
    return acc

test_loader = DataLoader(testset, batch_size=1000, shuffle=True)
acc_list = []
for test_inputs, test_labels in iter(test_loader):
    test_acc = get_acc(model, test_inputs, test_labels)
    acc_list.append(test_acc)
    
print("ACC", sum(acc_list)/len(acc_list))
# %%
