"""
Reward on expectation (over all goal and agent positions) achieved by an oracle
"""

import numpy as np


LABSIZE = 9
eplen = 200
rew = 10

lab = np.ones((LABSIZE, LABSIZE))
CTR = LABSIZE // 2  # default (4,4) center

# Grid maze
n_columns, n_available_pos = 0, 0
lab[1:LABSIZE - 1, 1:LABSIZE - 1].fill(0)
for row in range(1, LABSIZE - 1):
    for col in range(1, LABSIZE - 1):
        if row % 2 == 0 and col % 2 == 0:
            lab[row, col] = 1
            n_columns += 1
        else:
            n_available_pos += 1


mean_distance_to_this_lab = np.zeros((LABSIZE, LABSIZE))
#loop over each position where reward could be
for row in range(1, LABSIZE - 1):
    for col in range(1, LABSIZE - 1):
        # loop over each position where agent could be
        if lab[row, col] == 1:
            continue
        else:
            for other_row in range(1, LABSIZE - 1):
                for other_col in range(1, LABSIZE - 1):
                    if lab[other_row, other_col] == 1:
                        continue
                    else:
                        mean_distance_to_this_lab[row, col] += np.sum(np.abs(np.array([row-other_row, col-other_col])))

mean_distance_to_this_lab /= n_available_pos
print("mean_distance_to_this_lab",mean_distance_to_this_lab)

expected_reward_omniscient_agent = rew * eplen / mean_distance_to_this_lab
# remove inf from colums and walls
for row in range(0, LABSIZE):
    for col in range(0, LABSIZE):
        if lab[row, col] == 1:
            expected_reward_omniscient_agent[row, col] = 0

print("expected_reward_omniscient_agent per grid", expected_reward_omniscient_agent)
print('\n')
print("expected_reward_omniscient_agent", expected_reward_omniscient_agent.sum()/n_available_pos)



