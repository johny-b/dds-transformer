# %%
from typing import Callable

from endplay.dealer import generate_deal
from endplay.dds import solve_board
from endplay.types import Deal, Card, Denom, Player
from endplay._dds import DDSError  

# %%
def get_optimal_play(
    d: Deal, 
    card_selector: Callable[[list[Card]], Card],
) -> list[Card]:
    play = []
    while True:
        try:
            cards = [x[0] for x in solve_board(d)]
        except DDSError:
            break
        card = card_selector(cards)
        play.append(card)
        d.play(card)
    return play

def lowest_card(cards: list[Card]) -> Card:
    cards = sorted(cards, key=lambda x: (x.suit.value, x.rank.value))
    return cards[0]

def first_card(cards: list[Card]) -> Card:
    return cards[0]

def print_play(cards: list[Card | str]) -> None:
    for round_ix in range(len(cards) // 4):
        base_ix = round_ix * 4
        round_cards = cards[base_ix:base_ix + 4]
        round_cards_str = " ".join([str(c) for c in round_cards])
        print(round_cards_str)

# %%
def get_deal_data(
    suit: str, 
    leader: str,
):
    deal = generate_deal()
    deal.trump = Denom.find(suit)
    deal.first = Player.find(leader)
    
    # deal.pprint()
    
    deal_2 = deal.copy()

    data = [
        str(deal.first.abbr),
        str(deal.trump.abbr),
        str(deal),
    ]

    play_1 = get_optimal_play(deal, lowest_card)
    play_2 = get_optimal_play(deal_2, first_card)
    play_1 = [str(card) for card in play_1]
    play_2 = [str(card) for card in play_2]
    data.append(play_1)
    data.append(play_2)
    
    # print_play(play_1)
    
    
    return data

# get_deal_data("nt", "W")
# print()

# %%
