# %%
import ray

from generate_ending_board import get_deal_data

FILE_BOARDS = 10000
NUM_FILES = 300
NUM_CARDS = 5

# %%
@ray.remote
def get_board():
    data = get_deal_data(NUM_CARDS, "NT")
    data = [str(x) for x in data]
    return ";".join(data)

# %%
result_ids = [get_board.remote() for _ in range(NUM_FILES * FILE_BOARDS)]

file_ix = 0
while result_ids:
    done_ids, result_ids = ray.wait(result_ids, num_returns=FILE_BOARDS)
    data = ray.get(done_ids)
    data = "\n".join(data)
    fname = f'boards_{NUM_CARDS}_card_wt/{file_ix}.csv'
    with open(fname, 'w') as f:
        f.write(data)
    print(fname)
    file_ix += 1

# %%
