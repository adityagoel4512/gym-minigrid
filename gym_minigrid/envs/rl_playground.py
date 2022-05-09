from collections import namedtuple

import torch
from const import *
from gym_minigrid.minigrid import *

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class SafeExplorationEnv(MiniGridEnv):
    """
    Environment with wall or lava obstacles, sparse reward.
    """

    def __init__(self, size=9, lava_setup='', initial_state=(3, 2), max_steps=50, offline_regions=False,
                 rand_choices=3):
        self.initial_state = initial_state
        self.statistics = dict(lava_count=0)
        self.offline_regions = offline_regions
        self.statistics_arr = dict(lava_count=[])
        self.pause_stats = False
        lava_setups = {
            'corner': self.corner_lava,
            'wall': self.wall_lava,
            'none': lambda width, height, y_wall, end_x_wall: ()
        }

        self.lava_setup = lava_setups[lava_setup]
        self.rand_choices = rand_choices

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=None
        )

    def pause_statistics(self):
        self.pause_stats = True

    def all_cells(self, top_to_bottom=False):
        for row in range(self.height - 1, -1, -1) if top_to_bottom else range(self.height):
            for col in range(self.width):
                yield col, row, self.grid.get(col, row)

    def offline_cells(self):
        for col, row, cell in self.all_cells():
            if cell is None or cell.type == 'goal':
                yield col, row, cell

    def setup_offline_regions(self):
        for col, row, _ in self.offline_cells():
            self.grid.set(col, row, Floor(color='blue'))

    # def setup_offline_regions(self):
    #     for col, row, _ in self.offline_cells():
    #         state_vector = self.construct_state_vector((col, row), 0)
    #         if state_vector.squeeze()[1] < 0:
    #             self.grid.set(col, row, Floor(color='blue'))

    def corner_lava(self, width, height, y_wall, end_x_wall):
        self.wall_lava(width, height, y_wall, end_x_wall)

        corner_height = 3
        for c in range(corner_height):
            self.grid.vert_wall(width - c - 2, height - corner_height + c - 1, corner_height - c, Lava)
        # self.grid.vert_wall(end_x_wall+1, lava_y, 2, Lava)
        # for i in range(2):
        #     self.grid.horz_wall(1, y_wall+i+1, end_x_wall-i, Lava)

    def wall_lava(self, width, height, y_wall, end_x_wall):
        # Place lava (to measure safety of training)
        self.grid.horz_wall(1, y_wall + 1, end_x_wall, Lava)
        self.grid.horz_wall(1, y_wall, end_x_wall, Lava)
        self.grid.horz_wall(1, y_wall - 1, end_x_wall, Lava)
        # self.grid.vert_wall(width-2, 1, height-2, Lava)

    def _gen_grid(self, width, height, state=None, dir=None):
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height, Wall)

        # Generate barrier wall
        end_x_wall = 1 * width // 2 - 1
        y_wall = height // 2
        # self.grid.horz_wall(1, y_wall+1, end_x_wall, Wall)
        # self.grid.horz_wall(1, y_wall, end_x_wall, Wall)

        # choices = np.linspace(3, width-3, num=self.rand_choices, dtype=np.int)
        option_x = width // 3

        # Place the agent in the top-left corner
        self.agent_pos = np.array(state or (option_x, self.initial_state[1]))
        self.agent_dir = dir if dir is not None else np.random.randint(0, high=4)

        self.goal_state = np.array([3, height - 4])
        self.put_obj(Goal(), self.goal_state[0], self.goal_state[1])
        self.lava_setup(width, height, y_wall, end_x_wall)
        if self.offline_regions:
            self.setup_offline_regions()
        self.mission = (
            "reach the green goal square, dealing with obstacles and lava" if not self.offline_regions else "offline data regions"
        )

    def construct_state_vector(self, agent_pos, agent_dir):
        # center units and concatentate
        return np.array([agent_pos[0] - self.width // 2, agent_pos[1] - self.height // 2, agent_dir - 1])

    def deconstruct_state_vector(self, state_vector):
        if state_vector.shape[0] == 1:
            state_vector = state_vector.squeeze()
        pos = np.array([state_vector[0] + (self.width // 2), state_vector[1] + (self.height // 2)], dtype=np.int16)
        dir = state_vector[2].item() + 1
        return pos, dir

    def step(self, action):

        # Get the contents of the cell in front of the agent
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        obs, reward, done, info = super().step(action)
        reward = 0
        step_cost = 1
        if fwd_cell is not None and action == self.actions.forward:
            if fwd_cell.type == 'lava':
                # reward = -100 - 2.1*(self.max_steps - self.step_count)
                reward = -self.max_steps * STEP_COST
                if not self.pause_stats:
                    self.statistics['lava_count'] += 1
            elif fwd_cell.type == 'goal':
                reward = STEP_COST * self.max_steps * 1.5
            elif fwd_cell.type == 'wall':
                reward = -step_cost

        if not self.pause_stats:
            self.statistics_arr['lava_count'].append(self.statistics['lava_count'])

        info = dict(state_vector=self.construct_state_vector(self.agent_pos, self.agent_dir), **info)
        return info['state_vector'], MULTIPLIER * reward, done, info

    def reset(self):
        self.pause_stats = False
        return super().reset(), dict(state_vector=self.construct_state_vector(self.agent_pos, self.agent_dir))

    def half_offline_data(self):
        steps = self.step_count
        for transition in self.transitions_for_offline_data():
            if transition.state.squeeze()[1] < 0:
                yield transition
        self.step_count = steps

    def transitions_for_offline_data(self, extra_data=False, include_lava_actions=False, exclude_lava_neighbours=False,
                                     n_step=1, cut_step_cost=False, GAMMA=OFFLINE_GAMMA):
        self.pause_statistics()
        steps = self.step_count

        def neighbour_state_lava(col, row):
            for n_col, n_row in [(col - 1, row), (col + 1, row), (col, row - 1), (col, row + 1)]:
                cell = self.grid.get(n_col, n_row)
                if cell is not None and cell.type == 'lava':
                    return True
            return False

        def replicate(g, n, res):
            if n == 0:
                yield [n for n in res]
            else:
                for elem in g:
                    res.append(elem)
                    yield from replicate(g, n - 1, res)
                    res.pop()

        for col, row, cell in self.offline_cells():
            if exclude_lava_neighbours and neighbour_state_lava(col, row):
                continue
            print(col, row, cell)
            for dir in range(4):
                for action_sequence in replicate((self.actions.forward, self.actions.left, self.actions.right), n_step,
                                                 []):
                    print(action_sequence)
                    self._gen_grid(self.width, self.height, (col, row), dir)
                    state_vector = torch.from_numpy(
                        np.concatenate(self.construct_state_vector(self.agent_pos, self.agent_dir),
                                       axis=None)).float().unsqueeze(0)
                    action_vector = torch.tensor([[action_sequence[0]]], requires_grad=False)
                    cumulative_reward = 0
                    for i, action in enumerate(action_sequence):
                        fwd_cell = self.grid.get(*self.front_pos)
                        if include_lava_actions or (
                                self.actions.forward != action or fwd_cell is None or fwd_cell.type != 'lava'):
                            _, reward, _, done = self.step(action)
                            if cut_step_cost:
                                cumulative_reward += (reward - STEP_COST) * (GAMMA ** i)
                            else:
                                cumulative_reward += reward * (GAMMA ** i)
                            if self.actions.forward == action and fwd_cell is not None and fwd_cell.type == 'lava':
                                print('lava action!')
                            elif self.actions.forward == action and fwd_cell is not None and fwd_cell.type == 'goal':
                                print('goal action!')

                            if done or i == (len(action_sequence) - 1):
                                next_state_trace = torch.from_numpy(
                                    np.concatenate(self.construct_state_vector(self.agent_pos, self.agent_dir),
                                                   axis=None)).float().unsqueeze(0)
                                reward = torch.tensor([[cumulative_reward]], requires_grad=False)
                                print(Transition(state=state_vector, action=action_vector, reward=reward,
                                                 next_state=next_state_trace, done=done))
                                yield Transition(state=state_vector, action=action_vector, reward=reward,
                                                 next_state=next_state_trace, done=done)
                                break
                        else:
                            print('')
                            print('lava state + action')
                            print(state_vector)
                            print(action)
