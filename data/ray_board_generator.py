# %%
import ray
from datetime import datetime

from generate_ending_board import get_deal_data

NUM_CARDS = 13
TASK_BOARDS = 100
FILE_TASKS = 100

# %%
@ray.remote
def get_boards():
    out = []
    for x in range(TASK_BOARDS):
        data = get_deal_data(NUM_CARDS, "NT")
        data = [str(x) for x in data]
        out.append(";".join(data))
    return out

# %%
result_ids = [get_boards.remote() for _ in range(3 * FILE_TASKS)]

file_ix = 0
while result_ids:
    done_ids, result_ids = ray.wait(result_ids, num_returns=FILE_TASKS)
    data = ray.get(done_ids)
    flat_data = [row for rows in data for row in rows]
    result = "\n".join(flat_data)
    fname = f'boards_{NUM_CARDS}_full/{file_ix}.csv'
    with open(fname, 'w') as f:
        f.write(result)
    print(datetime.now(), fname)
    file_ix += 1
    result_ids += [get_boards.remote() for _ in range(FILE_TASKS)]


# %%
