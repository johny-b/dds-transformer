# %%
import torch as t
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
import ray
import numpy as np
import multiprocessing

from utils import pbn_to_repr, trick_to_repr, correct_cards_to_repr

@ray.remote
def _get_full_file_data(num_cards, file_ix):
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

# %%
class Dataset(TorchDataset):
    def __init__(self, card_files):
        self.card_files = card_files
        self.inputs, self.labels, self.num_tricks = self._get_data()
        
    def _get_data(self):
        num_cpus = 7
        ray.init(num_cpus=num_cpus)
        try:
            for num_cards, file_ids in self.card_files.items():
                print(f"Processing {num_cards}-card boards")
                result_ids = [_get_full_file_data.remote(num_cards, file_id) for file_id in file_ids]
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
        in_ = self.inputs[example_ix]
        out = self.labels[example_ix]
        tricks = self.num_tricks[example_ix]
        return self._rotate(in_, out, rotation_ix) + (tricks,)
    
    def _rotate(self, in_, out, rotation_ix):
        in_ = t.cat([self._rotate_hand(in_[i * 52: (i + 1) * 52], rotation_ix) for i in range(7)])
        out = self._rotate_hand(out, rotation_ix)
        return in_, out
    
    def _rotate_hand(self, hand, ix):
        assert ix in (0, 1, 2, 3)
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

# %%
