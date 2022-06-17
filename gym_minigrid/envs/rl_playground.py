import copy
from collections import namedtuple, defaultdict
from enum import IntEnum, Enum
import os

import gym
import numpy as np
import torch
from const import *
from gym_minigrid.minigrid import *

from minigrid.gym_minigrid.minigrid import MiniGridEnv

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class TerminationCondition(Enum):
    LAVA = 'LAVA'
    GOAL = 'GOAL'
    MAX_STEPS = 'MAX_STEPS'

class SafeExplorationEnv(MiniGridEnv):
    """
    Environment with wall or lava obstacles, sparse reward.
    """

    def __init__(self, size=9, lava_setup='', max_steps=50, offline_regions=False,
                 rand_choices=3, device=torch.device('cuda'), sparse_reward=True, ENV_CACHING = True):
        self.offline_regions = offline_regions
        self.statistics_arr = dict(lava_count=[0], terminalstates=[], goal=[0])
        self.pause_stats = False
        self.ENV_CACHING = ENV_CACHING
        lava_setups = {
            'v0': self.wall_lava,
            'v1': self.corner_lava,
            'v2': self.maze_lava
        }
        init_states = {
            'v0': (3, 2),
            'v1': (3, 3),
            'v2': (21, 16),
        }
        unc_penalised_rewards = {
            'v0': -5.765 if sparse_reward else -50,
            'v1': -2
        }

        step_costs = {
            'v0': 0.01, 
            'v1': 0.01,
        }

        multipliers = {
            'v0': 0.1,
            'v1': 0.1
        }

        self.MULTIPLIER = multipliers[lava_setup]
        self.STEP_COST = step_costs[lava_setup] # changed this in this run

        self.UNC_PENALISED_REWARD = unc_penalised_rewards[lava_setup]

        self.initial_state = init_states[lava_setup]
        self.device = device
        self.lava_choice = lava_setup
        self.lava_setup = lava_setups[lava_setup]
        self.rand_choices = rand_choices
        self.sparse_reward = sparse_reward
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

    def maze_lava(self, width, height, y_wall, end_x_wall):
        self.goal_state = np.array([3, 4])
        paths = {(4, 4), (5, 4), (6, 4), (7, 4), (8, 4), (9, 4), (10, 4), (11, 4), (12, 4),
                (4, 3), (5, 3), (6, 3), (7, 3), (8, 3), (9, 3), (10, 3), (11, 3), (12, 3),
                (11, 5), (12, 5),
                (11, 6), (12, 6),
                (11, 7), (12, 7),
                (11, 8), (12, 8),
                (11, 9), (12, 9), (13, 9), (14, 9), (15, 9), (16, 9),
                (11, 10), (12, 10), (13, 10), (14, 10), (15, 10), (16, 10),
                (15, 11), (16, 11),
                (15, 12), (16, 12),
                (15, 13), (16, 13),
                (15, 14), (16, 14),
                (11, 15), (12, 15), (13, 15), (14, 15), (15, 15), (16, 15),
                (11, 16), (12, 16), (13, 16), (14, 16), (15, 16), (16, 16),
                (3, 16), (4, 16), (5, 16), (6, 16), (7, 16), (8, 16), (9, 16), (10, 16),
                (11, 17), (12, 17), (13, 17), (14, 17), (15, 17), (16, 17),
                (3, 17), (4, 17), (5, 17), (6, 17), (7, 17), (8, 17), (9, 17), (10, 17),
                (10, 15), (9, 15), (8, 15), (3, 4), (21, 15), (7, 15),
                (7, 12), (7, 11), (8, 11), (9, 11), (9, 10), (6, 15), (5, 15), (4, 15), (4, 14),
                (6, 14),
                (5, 14), (5, 13), (5, 12), (5, 11), (5, 10), (5, 9), (5, 8), (5, 7), (5, 6), (5, 5),
                (4, 13), (4, 12), (4, 11), (4, 10), (4, 9), (4, 8), (4, 7), (4, 6), (4, 5),
                (3, 15), (3, 14), (3, 13), (3, 12), (3, 11), (3, 10), (3, 9), (3, 8), (3, 7), (3, 6), (3, 5),
                (2, 16), (2, 15), (2, 14), (2, 13), (2, 12), (2, 11), (2, 10), (2, 9), (2, 8), (2, 7), (2, 6), (2, 5),
                (2, 4), (2, 3), (3, 3), (4, 3)}

        for r in range(1, height-1):
            for c in range(1, width-1):
                if (c, r) not in paths:
                    self.put_obj(Lava(), c, r)

    def corner_lava(self, width, height, y_wall, end_x_wall):

        corner_height = 3
        for c in range(corner_height):
            self.grid.vert_wall(width - c - 2, height - corner_height + c - 1, corner_height - c, Lava)
        # self.grid.vert_wall(end_x_wall+1, lava_y, 2, Lava)
        # for i in range(2):
        #     self.grid.horz_wall(1, y_wall+i+1, end_x_wall-i, Lava)
        self.put_obj(Lava(), 1, 1)
        self.put_obj(Lava(), 1, 2)
        self.put_obj(Lava(), 1, 3)
        self.put_obj(Lava(), 2, 2)
        self.put_obj(Lava(), 2, 3)

        self.put_obj(Lava(), width-5, 2)
        self.put_obj(Lava(), width-4, 3)
        self.put_obj(Lava(), width-3, 4)
        self.put_obj(Lava(), width-2, 5)
        self.put_obj(Lava(), 4, height-2)
        self.put_obj(Lava(), width//2, height//2)
        self.put_obj(Lava(), 4, height-5)
        self.put_obj(Lava(), 3, 4)

        self.put_obj(Lava(), 4, height//2 + 1)
        self.put_obj(Lava(), 5, height//2 + 1)
        self.put_obj(Lava(), 3, height//2 + 2)
        self.put_obj(Lava(), 4, height//2 + 2)
        self.put_obj(Lava(), 5, height//2 + 2)
        self.put_obj(Lava(), 6, height//2 + 2)
        self.put_obj(Lava(), 2, height//2 + 3)
        self.put_obj(Lava(), 3, height//2 + 3)
        self.put_obj(Lava(), 4, height//2 + 3)
        self.put_obj(Lava(), 5, height//2 + 3)
        self.put_obj(Lava(), 6, height//2 + 3)
        self.put_obj(Lava(), 3, height//2 + 4)
        self.put_obj(Lava(), 4, height//2 + 4)
        self.put_obj(Lava(), 4, height//2 + 5)
        self.put_obj(Lava(), 4, height//2 + 8)
        self.put_obj(Lava(), 4, height - 4)

        self.put_obj(Lava(), width-7, height-4)
        self.put_obj(Lava(), width//2 - 3, height-8)
        self.goal_state = np.array([width-5, height - 6])

    def wall_lava(self, width, height, y_wall, end_x_wall):
        # Place lava (to measure safety of training)
        self.grid.horz_wall(1, y_wall + 1, end_x_wall, Lava)
        self.grid.horz_wall(1, y_wall, end_x_wall, Lava)
        self.grid.horz_wall(1, y_wall - 1, end_x_wall, Lava)
        self.goal_state = np.array([3, height - 4])
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
        self.agent_pos = np.array(state or (option_x, self.initial_state[1]), dtype=float)
        self.agent_dir = dir if dir is not None else np.random.randint(0, high=4)
        if isinstance(self, ContinuousSafeExplorationEnv):
            self.agent_dir = np.random.uniform(low=-np.pi, high=np.pi)
            print(self.agent_dir)
        elif isinstance(self, DiscreteSafeExplorationEnv):
            pass

        self.step_count = 0

        self.lava_setup(width, height, y_wall, end_x_wall)
        self.put_obj(Goal(), self.goal_state[0], self.goal_state[1])
        if self.offline_regions:
            self.setup_offline_regions()
        self.mission = (
            "Reach the green goal square, dealing with obstacles and lava" if not self.offline_regions else "offline data regions"
        )
        self.done = False

    def construct_state_vector(self, agent_pos, agent_dir):
        # center units and concatentate
        return np.array([agent_pos[0] - self.width // 2, agent_pos[1] - self.height // 2, agent_dir - 1])

    def deconstruct_state_vector(self, state_vector):
        state_vector = state_vector.cpu()
        if state_vector.shape[0] == 1:
            state_vector = state_vector.squeeze()
        pos = np.array([state_vector[0] + (self.width // 2), state_vector[1] + (self.height // 2)], dtype=np.int16)
        dir = state_vector[2].item() + 1
        return pos, dir

    def transitions_for_offline_data(self, extra_data=False, include_lava_actions=False, exclude_lava_neighbours=False,
                                     n_step=1, cut_step_cost=False, GAMMA=OFFLINE_GAMMA, deduplicate=True):

        arg_string = ':'.join(str(u) for u in (extra_data, include_lava_actions, exclude_lava_neighbours, n_step, cut_step_cost, GAMMA, deduplicate))
        cache_path = f'{OUTPUT_LOCATION}/datasetcache/{self.lava_choice}:{arg_string}.pkl'
        print(f'deduplicate: {deduplicate}')
        if os.path.exists(cache_path) and os.path.isfile(cache_path):
            print(f'loading from cache: {cache_path}')
            dataset = torch.load(cache_path)
            shift_transitions = dataset[0].state.device != self.device
            if not shift_transitions:
                for transition in dataset:
                    if transition.done:
                        if transition.reward > 0:
                            print(f'goal: {transition}')
                        else:
                            print(f'lava: {transition}')
                    yield transition
            else:
                print(f'shifting data into: {self.device}')
                for transition in dataset:
                    if transition.done:
                        if transition.reward > 0:
                            print(f'goal: {transition}')
                        else:
                            print(f'lava: {transition}')
                    t = Transition(state=transition.state.to(self.device), action=transition.action.to(self.device), next_state=transition.next_state.to(self.device), reward=transition.reward.to(self.device), done=transition.done.to(self.device))
                    yield t
        else:
            dataset = list(self.__transitions_for_offline_data(extra_data, include_lava_actions, exclude_lava_neighbours, n_step, cut_step_cost, GAMMA))
            print(f'cache write path: {cache_path}')
            if deduplicate:
                print(f'deduplicating ...')
                nonduplicates = []
                nonduplicate_dict = defaultdict(list)
                def find(t):
                    for tr in nonduplicate_dict[(round(t.state.flatten()[0].item()), round(t.state.flatten()[1].item()))]:
                        if equal_transition(t, tr, False):
                            return True
                    return False
                for transition in dataset:
                    if not find(transition):
                        nonduplicate_dict[(round(transition.state.flatten()[0].item()), round(transition.state.flatten()[1].item()))].append(transition)
                        nonduplicates.append(transition)
                    else:
                        print(f'{transition} is duplicate')
                if self.ENV_CACHING:
                    print(f'before length: {len(dataset)}, after: {len(nonduplicates)}')
                    torch.save(nonduplicates, cache_path)
                    yield from self.transitions_for_offline_data(extra_data, include_lava_actions, exclude_lava_neighbours, n_step, cut_step_cost, GAMMA)
                else:
                    for transition in nonduplicates:
                        yield transition
            elif self.ENV_CACHING:
                torch.save(dataset, cache_path)
                yield from self.transitions_for_offline_data(extra_data, include_lava_actions, exclude_lava_neighbours, n_step, cut_step_cost, GAMMA)
            else:
                for transition in dataset:
                    yield transition



    def __transitions_for_offline_data(self, extra_data=False, include_lava_actions=False, exclude_lava_neighbours=False,
                                     n_step=1, cut_step_cost=False, GAMMA=OFFLINE_GAMMA):
        
        self.pause_statistics()

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
                for action_sequence in replicate(self.actions, n_step,
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
                            _, reward, done, _ = self.step(action)
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
                                # if done and not include_post_terminal_transitions:
                                #     next_state_trace = None
                                if i == (len(action_sequence)-1) and n_step > 1:
                                    done = False
                                transition = Transition(state=state_vector.to(self.device), action=action_vector.to(self.device), reward=reward.to(self.device),
                                                        next_state=next_state_trace.to(self.device), done=torch.tensor([[done]]).to(self.device))
                                print(transition)
                                yield transition
                                break
                        else:
                            print('')
                            print('lava state + action')
                            print(state_vector)
                            print(action)


class DiscreteActions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2

class DiscreteSafeExplorationEnv(SafeExplorationEnv):

    def __init__(self, **kwargs):
        super(DiscreteSafeExplorationEnv, self).__init__(**kwargs)
        self.actions = DiscreteActions
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.done = False
        print(f'sparse reward: {self.sparse_reward}')

    def step(self, action):
        if self.done:
            # print('DiscreteSafeExplorationEnv is done but step invoked. Returning')
            return
        # Get the contents of the cell in front of the agent
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        obs, reward, done, info = super().step(action)
        reward = 0
        if done:
            info['reason'] = f'Max Steps at {self.agent_pos}'
            info['termination'] = TerminationCondition.MAX_STEPS
            if not self.pause_stats:
                self.statistics_arr['goal'].append(self.statistics_arr['goal'][-1])
                self.statistics_arr['lava_count'].append(self.statistics_arr['lava_count'][-1])

        if fwd_cell is not None and action == self.actions.forward:
            if fwd_cell.type == 'lava':
                # reward = -100 - 2.1*(self.max_steps - self.step_count)
                reward = -self.max_steps * self.STEP_COST
                info['reason'] = f'Lava at {self.agent_pos}'
                info['termination'] = TerminationCondition.LAVA
                if not self.pause_stats:
                    self.statistics_arr['lava_count'][-1] += 1
            elif fwd_cell.type == 'goal':
                reward = self.STEP_COST * self.max_steps * 2
                print(reward)
                info['reason'] = f'Goal at {self.agent_pos}'
                info['termination'] = TerminationCondition.GOAL
                if not self.pause_stats:
                    self.statistics_arr['goal'][-1] += 1
            elif fwd_cell.type == 'wall':
                reward = -1
        elif np.sqrt(np.dot(self.agent_pos-self.goal_state, self.agent_pos-self.goal_state)) < 0.5:
            done = True
            reward = self.STEP_COST * self.max_steps * 2
            print(reward)
            info['reason'] = f'Goal at {self.agent_pos}'
            info['termination'] = TerminationCondition.GOAL
            if not self.pause_stats:
                self.statistics_arr['goal'][-1] += 1

        if not self.sparse_reward or self.lava_choice == 'v1' or self.lava_choice == 'v2':
            reward += np.exp(-np.sqrt(np.dot(self.agent_pos-self.goal_state, self.agent_pos-self.goal_state))/5)*0.01
            reward += float((1. - np.sqrt(np.dot(self.agent_pos-self.goal_state, self.agent_pos-self.goal_state)) + self.agent_pos[1]) * self.STEP_COST * self.max_steps * 1.5) * 0.01

        # if self.lava_choice == 'v2':
        #     reward += 1.2
        info = dict(state_vector=self.construct_state_vector(self.agent_pos, self.agent_dir), **info)
        self.done = done
        return info['state_vector'], self.MULTIPLIER * reward, done, info

    def reset(self):
        self.pause_stats = False
        self.done = False
        return super().reset(), dict(state_vector=self.construct_state_vector(self.agent_pos, self.agent_dir))


class ContinuousSafeExplorationEnv(SafeExplorationEnv):
    def __init__(self, degree_mode=False, **kwargs):
        self.degree_mode = degree_mode
        super(ContinuousSafeExplorationEnv, self).__init__(**kwargs)

        # action_space is a 2d action space
        # the first continuous value represents the orientation turned (bounded in range -pi to pi)
        # positive rotation = rotating right
        # negative rotation = rotating left
        # when orientation is 0, this corresponds to facing east
        # the second continuous value represents the distance travelled (clipped to 1 in step)

        self.forward_candidate_step = None
        print(f'degree mode: {degree_mode}')

        low = np.array([-180., 000.])
        high = np.array([180., 1])
        self.action_space = gym.spaces.Box(low=low, high=high)

        def slice_action_space():
            for rotation in np.linspace(low[0], high[0], num=4):
                for movement_forward in np.linspace(low[1], high[1], num=4):
                    print([rotation, movement_forward])
                    yield np.array([rotation, movement_forward])

        self.actions = slice_action_space()

        # always maintained in radians
        self.agent_dir = 0

    def rotate_agent(self, rotation):
        self.agent_dir = self.agent_dir + rotation
        while self.agent_dir < -np.pi:
            self.agent_dir += 2 * np.pi

        while self.agent_dir > np.pi:
            self.agent_dir -= 2 * np.pi

    def forward_movement_pos_change(self, distance):
        # assume distance in [0, 1]
        dx = distance * np.cos(self.agent_dir)
        dy = distance * np.sin(self.agent_dir)
        return dx, dy

    @property
    def front_pos(self):
        # front_pos in continuous scenario means the integer grid in front after moving forward 1 step
        dx, dy = self.forward_movement_pos_change(self.forward_candidate_step)
        front_pos = np.array((int(self.agent_pos[0] + dx), int(self.agent_pos[1] + dy)))
        return front_pos

    @staticmethod
    def degree_to_rad(deg):
        return deg * np.pi / 180

    @staticmethod
    def rad_to_degree(rad):
        return rad * 180 / np.pi

    def step(self, action):
        if action.ndim == 2 and action.shape[0] == 1:
            action = action.flatten()
        self.step_count += 1
        # print(f'Before Pos: {self.agent_pos}, Orientation: {self.agent_dir}')
        rotation = ContinuousSafeExplorationEnv.degree_to_rad(action[0])
        # action is 2d np array flattened
        self.rotate_agent(rotation)
        self.forward_candidate_step = np.clip(action[1], 0, 1)
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        if fwd_cell is None or fwd_cell.can_overlap():
            dx, dy = self.forward_movement_pos_change(self.forward_candidate_step)
            self.agent_pos[0] += dx
            self.agent_pos[1] += dy
            self.forward_candidate_step = None
            # successfully advance forward
            # print(f'successfully advanced: {dx, dy}')
        else:
            # print(f'did not advance')
            assert fwd_cell.type == 'wall', 'Non wall failure to move'
        # print(f'After Pos: {self.agent_pos}, Orientation: {self.agent_dir}')

        reward = 0
        done = self.step_count >= self.max_steps
        info = dict()
        if not self.pause_stats:
            self.statistics_arr['goal'].append(self.statistics_arr['goal'][-1])
            self.statistics_arr['lava_count'].append(self.statistics_arr['lava_count'][-1])
        dist = np.sqrt(np.dot(self.goal_state - self.agent_pos, self.goal_state - self.agent_pos))
        if fwd_cell is not None:
            if fwd_cell.type == 'lava':
                # reward = -100 - 2.1*(self.max_steps - self.step_count)
                reward = -self.max_steps * STEP_COST
                done = True
                if not self.pause_stats:
                    self.statistics_arr['lava_count'][-1] += 1
                info['reason'] = f'Lava at {self.agent_pos}'
            elif fwd_cell.type == 'goal' or dist < 0.5:
                reward = STEP_COST * self.max_steps * 2
                info['reason'] = f'Goal at {self.agent_pos}'
                done = True
                if not self.pause_stats:
                    self.statistics_arr['goal'][-1] += 1
            elif fwd_cell.type == 'wall':
                reward = -1
        elif dist < 0.5:
            reward = STEP_COST * self.max_steps * 2
            info['reason'] = f'Goal at {self.agent_pos}'
            done = True
            if not self.pause_stats:
                self.statistics_arr['goal'][-1] += 1

        reward += float((15. - np.sqrt(np.dot(self.agent_pos-self.goal_state, self.agent_pos-self.goal_state)) + self.agent_pos[1]) * STEP_COST * self.max_steps * 2) * 0.01

        if not self.pause_stats:
            if done:
                self.statistics_arr['terminalstates'].append(copy.deepcopy(self.agent_pos))

        info = dict(state_vector=self.construct_state_vector(self.agent_pos, self.agent_dir),
                    **info)
        return info['state_vector'], MULTIPLIER * reward, done, info

    def construct_state_vector(self, agent_pos, agent_dir):
        # center units and concatentate
        # agent_dir always in radians
        agent_dir = ContinuousSafeExplorationEnv.rad_to_degree(agent_dir)
        return np.array([agent_pos[0] - self.width // 2, agent_pos[1] - self.height // 2, agent_dir], dtype=float)

    def deconstruct_state_vector(self, state_vector):
        if state_vector.shape[0] == 1:
            state_vector = state_vector.squeeze()
        pos = np.array([state_vector[0] + float(self.width // 2), state_vector[1] + float(self.height // 2)], dtype=float)
        dir = state_vector[2].item()
        return pos, dir

    def reset(self):
        self.pause_stats = False
        self.forward_candidate_step = None
        reset = super().reset()
        self.agent_dir = 0
        return reset, dict(state_vector=self.construct_state_vector(self.agent_pos, self.agent_dir))
