# %%
import torch as t
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from utils import pbn_to_repr, trick_to_repr, correct_cards_to_repr

# %%
class Dataset(TorchDataset):
    def __init__(self, card_files):
        self.card_files = card_files
        print("Reading input files")
        self.pbn_data = self._get_pbn_data()
        print("Processing boards")
        self.data = self._process_boards()
        
    def _process_boards(self):
        data = []
        for pbn, current_trick, correct_cards, _ in tqdm(self.pbn_data):
            in_ = pbn_to_repr(pbn)
            trick_data = trick_to_repr(current_trick)
            out = correct_cards_to_repr(correct_cards)
            in_ = t.cat([in_, trick_data])
            data.append((in_, out))
        return data
        
    def __len__(self):
        #   4 for each rotation
        return len(self.data) * 4
    
    def __getitem__(self, ix):
        example_ix = ix % len(self.data)
        rotation_ix = ix // len(self.data)
        return self._rotate(self.data[example_ix], rotation_ix)
    
    def _rotate(self, in_out, rotation_ix):
        in_, out = in_out
        in_ = t.cat([self._rotate_hand(in_[i * 52: (i + 1) * 52], rotation_ix) for i in range(7)])
        out = self._rotate_hand(out, rotation_ix)
        return in_, out
    
    def _rotate_hand(self, hand, ix):
        assert ix in (0, 1, 2, 3)
        return t.cat([hand[13 * ix:], hand[:13 * ix]])
    
    def _get_pbn_data(self):
        data = []
        for num_cards, file_ids in self.card_files.items():
            for file_ix in file_ids:
                fname = f"data/boards_{num_cards}_card_wt_all/{file_ix}.csv"
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
        pbn, current_trick, correct_cards, tricks = parts[2], parts[3], parts[4], parts[5]
        
        return pbn, current_trick, correct_cards, tricks
    
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
        
        #   "N:" prefix
        if pbn[1] == ':':
            new_pbn = pbn[:2] + new_pbn

        return new_pbn

# %%
