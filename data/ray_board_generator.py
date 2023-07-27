# %%
import ray

from generate_ending_board import get_deal_data

FILE_BOARDS = 10000
NUM_FILES = 100
NUM_CARDS = 2

# %%
@ray.remote
def get_board():
    data = get_deal_data(NUM_CARDS, "NT", "N")
    data[4] = str(data[4])
    return ";".join(data)

# %%
result_ids = [get_board.remote() for _ in range(NUM_FILES * FILE_BOARDS)]

file_ix = 0
while result_ids:
    done_ids, result_ids = ray.wait(result_ids, num_returns=FILE_BOARDS)
    data = ray.get(done_ids)
    data = "\n".join(data)
    fname = f'boards_{NUM_CARDS}_card/{file_ix}.csv'
    with open(fname, 'w') as f:
        f.write(data)
    file_ix += 1
    
# %%
