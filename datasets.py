# %%
import torch as t
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
import ray
import numpy as np
import multiprocessing

from utils import pbn_to_repr, trick_to_repr, correct_cards_to_repr

@ray.remote
def _get_full_file_data_old(num_cards, file_ix):
    fname = f"data/boards_{num_cards}_card_wt_all/{file_ix}.csv"
    with open(fname, 'r') as f:
        rows = [row.strip() for row in f.readlines()]
    
    inputs, outputs, num_tricks = [], [], []
    for row in rows:
        pbn, current_trick, correct_cards, tricks = row.split(';')[2:6]
        tricks = int(tricks)
        in_ = pbn_to_repr(pbn)
        trick_data = trick_to_repr(current_trick)
        out = correct_cards_to_repr(correct_cards)
        in_ = t.cat([in_, trick_data])
        inputs.append(in_)
        outputs.append(out)
        num_tricks.append(tricks)
    return t.stack(inputs), t.stack(outputs), t.Tensor(num_tricks)

@ray.remote
def _get_full_file_data_new(num_cards, file_ix):
    fname = f"data/boards_{num_cards}_full/{file_ix}.csv"
    with open(fname, 'r') as f:
        rows = [row.strip() for row in f.readlines()]
    
    inputs, labels, num_tricks = [], [], []
    for row in rows:
        pbn, current_trick, *all_tricks = row.split(';')[1:]
        in_ = pbn_to_repr(pbn)
        trick_data = trick_to_repr(current_trick)
        in_ = t.cat([in_, trick_data])
        inputs.append(in_)
        
        label_parts = []
        for tricks in all_tricks:
            tricks = [int(x) for x in tricks.split(' ')]
            
            #   0/1 - do we have this card
            label = in_[:52].clone()
            card_ixs = label.nonzero().flatten()
            
            #   set nonexistent cards to -2
            label[label == 0] = -2
            
            #   set other values
            for card_ix, tr in zip(card_ixs, tricks):
                label[card_ix] = tr
            label_parts.append(label)

        full_label = t.cat(label_parts)
        labels.append(full_label)
        num_tricks.append(full_label.max().item())

    return t.stack(inputs), t.stack(labels), t.Tensor(num_tricks)
# %%
class Dataset(TorchDataset):
    def __init__(self, card_files, card_tricks=False):
        self.card_files = card_files
        self.card_tricks = card_tricks
        self.inputs, self.labels, self.num_tricks = self._get_data()
        
    def _get_data(self):
        num_cpus = 24
        ray.init(num_cpus=num_cpus)
        func = _get_full_file_data_new if self.card_tricks else _get_full_file_data_old
        try:
            for num_cards, file_ids in self.card_files.items():
                print(f"Processing {num_cards}-card boards")
                result_ids = [func.remote(num_cards, file_id) for file_id in file_ids]
                all_done_ids = []
                for _ in tqdm(file_ids):
                    new_done_ids, result_ids = ray.wait(result_ids, num_returns=1)
                    all_done_ids += new_done_ids
            all_data = ray.get(all_done_ids)
            inputs = [x[0] for x in all_data]
            outputs = [x[1] for x in all_data]
            num_tricks = [x[2] for x in all_data]
            return t.cat(inputs), t.cat(outputs), t.cat(num_tricks)
        finally:
            ray.shutdown()
        
    def __len__(self):
        #   4 for each rotation
        return len(self.inputs) * 4
    
    def __getitem__(self, ix):
        example_ix = ix % len(self.inputs)
        rotation_ix = ix // len(self.inputs)
        assert rotation_ix in (0, 1, 2, 3)
        in_ = self.inputs[example_ix]
        out = self.labels[example_ix]
        tricks = self.num_tricks[example_ix]
        return self._rotate(in_, out, rotation_ix) + (tricks,)
    
    def _rotate(self, in_, out, rotation_ix):
        in_ = t.cat([self._rotate_hand(in_[i * 52: (i + 1) * 52], rotation_ix) for i in range(7)])
        out = t.cat([self._rotate_hand(out[i * 52: (i + 1) * 52], rotation_ix) for i in range(5)])
        out = t.cat([out[:52], out[52 * (rotation_ix + 1):], out[:52 * (rotation_ix + 1)]])
        return in_, out
    
    def _rotate_hand(self, hand, ix):
        
        return t.cat([hand[13 * ix:], hand[:13 * ix]])    
    
    def _remove_card(self, pbn, card):
        # NOT USED NOW
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
        
        #   "N:" prefix
        if pbn[1] == ':':
            new_pbn = pbn[:2] + new_pbn

        return new_pbn

# # %%
# x = Dataset({13: [0]}, card_tricks=True)


# # %%
# print([int(a) for a in x[4][1][52:104]])
# print([int(a) for a in x[10004][1][52 * 5:]])
# # %%
