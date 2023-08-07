#   Generate ending board with
#   *   Given number of cards per player
#   *   Board states such that current player has
#       a single card that is better than others
# %%
from itertools import product
from random import shuffle, choice

from endplay.dealer import generate_deal
from endplay.dds import solve_board
from endplay.types import Deal, Denom, Player, Rank, Card



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

def play_random_card(deal: Deal):
    hand = deal.curhand
    cards = []
    if deal.curtrick:
        suit = deal.curtrick[0].suit
        cards = [Card(suit=suit, rank=rank) for rank in hand[suit]]

    if not cards:
        cards = list(iter(hand))
    random_card = choice(cards)
    deal.play(random_card)

def get_deal_data(player_cards_cnt: int, suit: str):
    #   Generate any random deal
    deal = generate_any_deal(player_cards_cnt)
    
    #   Play cards in a trick
    cards_played = choice([0,1,2,3])
    players = "NWSE"
    deal.first = Player.find(players[cards_played])
    for i in range(cards_played):
        play_random_card(deal)

    assert deal.curplayer == Player.find("N")
    
    
    out = [
        str(deal.curplayer.abbr),
        str(deal),
        " ".join(str(card) for card in deal.curtrick),
    ]

    #   Calculate results for all denoms
    for trump in (Denom.nt, Denom.spades, Denom.hearts, Denom.diamonds, Denom.clubs):
        deal.trump = trump
        solved_board = solve_board(deal)
        
        def s(x: tuple[Card, int]):
            return x.suit.numerator * 100000 - x.rank.numerator
        
        my_cards = sorted(deal[0], key=s)
        cards_with_tricks = list(solved_board)
        
        tricks = []
        for card in my_cards:
            for other_card, tr in cards_with_tricks:
                if other_card == card:
                    break
            else:
                tr = -1
            tricks.append(tr)

        out.append(" ".join([str(x) for x in tricks]))

    return out

# %%
