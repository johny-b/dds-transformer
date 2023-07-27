# %%
import torch as t

# %%
def pbn_to_repr(pbn):    
    #   Returns bool tensor of length 204
    if pbn[1] == ':':
        pbn = pbn[2:]
    hands = pbn.split(' ')
    result = t.cat([parse_pbn_hand(hand) for hand in hands])
    assert len(result) == 208, f"Incorrect pbn: {pbn}"
    return result

def parse_pbn_hand(hand):
    suits = hand.split('.')
    return t.cat([parse_pbn_suit(suit) for suit in suits])

def parse_pbn_suit(suit):
    cards = list(str(x) for x in range(2, 10)) + list('TJQKA')
    cards.reverse()
    has_card = [card in suit for card in cards]
    return t.tensor(has_card, dtype=t.float32)
# %%
def repr_to_pbn(full_repr):
    hands = [
        hand_repr_to_pbn(full_repr[:52]),
        hand_repr_to_pbn(full_repr[52:104]),
        hand_repr_to_pbn(full_repr[104:156]),
        hand_repr_to_pbn(full_repr[156:]),
    ]
    return " ".join(hands)

def hand_repr_to_pbn(hand_repr):
    suits = [
        suit_repr_to_pbn(hand_repr[:13]),
        suit_repr_to_pbn(hand_repr[13:26]),
        suit_repr_to_pbn(hand_repr[26:39]),
        suit_repr_to_pbn(hand_repr[39:]),
    ]
    return ".".join(suits)

def suit_repr_to_pbn(suit_repr):
    cards = list(str(x) for x in range(2, 10)) + list('TJQKA')
    cards.reverse()
    suit = [cards[x] for x in range(13) if suit_repr[x]]
    return "".join(suit)
# %%

pbn = 'A3.A.. KQ... JT... 98...'
in_ = pbn_to_repr(pbn)
assert repr_to_pbn(in_) == pbn

# %%
