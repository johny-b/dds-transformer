#   Generate ending board with
#   *   Given number of cards per player
#   *   Board states such that current player has
#       a single card that is better than others
# %%
from itertools import product
from random import shuffle

from endplay.dealer import generate_deal
from endplay.dds import solve_board
from endplay.types import Deal, Denom, Player, Rank



# %%
def cards_2_pbn(cards):
    parts = []
    for suit in (Denom.spades, Denom.hearts, Denom.diamonds, Denom.clubs):
        suit_cards = [card[0].abbr for card in cards if card[1] == suit]
        parts.append("".join(sorted(suit_cards, reverse=True)))
    return ".".join(parts)

def generate_any_deal(player_cards_cnt):
    """Just a random Deal with this many cards per player"""
    all_cards = list(product(list(Rank), [x for x in Denom if x.is_suit()]))
    assert len(all_cards) == 52
    
    total_cards_cnt = player_cards_cnt * 4
    shuffle(all_cards)
    selected_cards = all_cards[:total_cards_cnt]
    
    hands = []
    for i in range(4):
        hand_cards = selected_cards[i * player_cards_cnt: (i + 1) * player_cards_cnt]
        hand_pbn = cards_2_pbn(hand_cards)
        hands.append(hand_pbn)
    
    return Deal(" ".join(hands))
    
    
def get_deal_data(player_cards_cnt: int, suit: str, leader: str):
    while True:
        deal = generate_any_deal(player_cards_cnt)
        deal.trump = Denom.find(suit)
        deal.first = Player.find(leader)            
        out = [
            str(deal.first.abbr),
            str(deal.trump.abbr),
            str(deal),
        ]

        data = list(solve_board(deal))
        if len(data) == 1:
            card = data[0][0]
            tricks = data[0][1]
            out.append(str(card))
            out.append(tricks)
            return out

# %%
