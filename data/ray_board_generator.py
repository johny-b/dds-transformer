# %%
import ray

from generate_board import get_deal_data

NUM_BOARDS = 10000
FILE_BOARDS = 1000

# %%
@ray.remote
def get_board():
    data = get_deal_data("NT", "W")
    data[3] = "".join(data[3])
    data[4] = "".join(data[4])
    return ";".join(data)

# %%
result_ids = [get_board.remote() for _ in range(NUM_BOARDS)]

file_ix = 0
while result_ids:
    done_ids, result_ids = ray.wait(result_ids, num_returns=FILE_BOARDS)
    data = ray.get(done_ids)
    data = "\n".join(data)
    fname = f'boards/{file_ix}.csv'
    with open(fname, 'w') as f:
        f.write(data)
    file_ix += 1
    
# %%
