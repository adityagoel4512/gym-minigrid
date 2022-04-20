from operator import ne

from zmq import device
from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import copy
import torch
from collections import namedtuple

Transition = namedtuple('Transitions', ('state', 'action', 'reward', 'next_state'))
class SafeExplorationEnv(MiniGridEnv):
    """
    Environment with wall or lava obstacles, sparse reward.
    """

    def __init__(self, size=9, lava_setup='', initial_state=(3, 2), max_steps=50, reward_multiplier=0.1, offline_regions=False):
        self.initial_state = initial_state
        self.reward_multiplier = reward_multiplier
        self.statistics = dict(lava_count=0)
        self.offline_regions = offline_regions
        self.statistics_arr = dict(lava_count=[])
        lava_setups = {
            'corner': self.corner_lava,
            'wall': self.wall_lava,
            'none': lambda width, height, y_wall, end_x_wall: ()
        }
        
        self.lava_setup = lava_setups[lava_setup]

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=None
        )
    
    def offline_cells(self):
        for row in range(self.height):
            for col in range(self.width):
                cell = self.grid.get(col, row)
                if cell is None or cell.type == 'goal':
                    yield col, row, cell

    def setup_offline_regions(self):
        for col, row, _ in self.offline_cells():
            self.grid.set(col, row, Floor(color='blue'))
    
    def corner_lava(self, width, height, y_wall, end_x_wall):

        # Place lava (to measure safety of training)
        lava_y = y_wall-1
        
        
        # self.grid.vert_wall(width-2, 1, height-2, Lava)
        corner_height = 3
        for c in range(corner_height):
            self.grid.vert_wall(width-c-2, height-corner_height+c-1, corner_height-c, Lava)
        # self.grid.vert_wall(end_x_wall+1, lava_y, 2, Lava)
        # for i in range(2):
        #     self.grid.horz_wall(1, y_wall+i+1, end_x_wall-i, Lava)

    def wall_lava(self, width, height, y_wall, end_x_wall):
        # Place lava (to measure safety of training)
        self.grid.horz_wall(1, y_wall-1, end_x_wall, Lava)
        self.grid.horz_wall(1, height//2, end_x_wall, Lava)


    def _gen_grid(self, width, height, state=None, dir=None):
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Generate barrier wall
        end_x_wall = 11*width//20
        y_wall = height//2
        self.grid.horz_wall(1, y_wall-1, end_x_wall, Wall)
        self.grid.horz_wall(1, y_wall, end_x_wall, Wall)

        # Place the agent in the top-left corner
        self.agent_pos = np.array(state or self.initial_state)
        self.agent_dir = dir or 0

        # Place a goal square in the bottom-right corner
        self.goal_state = np.array([3, height-4])
        self.put_obj(Goal(), self.goal_state[0], self.goal_state[1])
        self.lava_setup(width, height, y_wall, end_x_wall)  
        if self.offline_regions:      
            self.setup_offline_regions()
        self.mission = (
            "reach the green goal square, dealing with obstacles and lava" if not self.offline_regions else "offline data regions"
        )
    
    def construct_state_vector(self, agent_pos, agent_dir):
        # center units and concatentate
        agent_pos = copy.deepcopy(agent_pos)
        agent_dir = copy.deepcopy(agent_dir)
        agent_pos[0] -= self.width//2
        agent_pos[1] -= self.height//2
        agent_dir -= 1
        return np.append(self.agent_pos, self.agent_dir)
    
    def deconstruct_state_vector(self, state_vector):
        pos = np.array([state_vector[0]*(self.width//2), state_vector[1]*(self.height//2)], dtype=np.int8)
        dir = state_vector[2] + 1
        return pos, dir

    def step(self, action):
        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)
        obs, reward, done, info = super().step(action)
        reward = 0
        step_cost = 1
        if action == self.actions.forward and fwd_cell != None and fwd_cell.type == 'lava':
            # reward = -100
            reward = -4000
            self.statistics['lava_count'] += 1
        elif action == self.actions.forward and fwd_cell != None and fwd_cell.type == 'goal':
            reward = step_cost*self.max_steps*1.25
        elif action == self.actions.forward and fwd_cell != None and fwd_cell.type == 'wall':
            reward = -step_cost
        
        if done:
            reward -= np.mean((self.agent_pos - self.goal_state)**2)
        else:
            reward -= 2.1*step_cost
        
        self.statistics_arr['lava_count'].append(self.statistics['lava_count'])
        info = dict(state_vector=self.construct_state_vector(self.agent_pos, self.agent_dir), **info)
        return obs, self.reward_multiplier * reward, done, info

    def reset(self):
        return super().reset(), dict(state_vector=self.construct_state_vector(self.agent_pos, self.agent_dir))

    def transitions_for_offline_data(self, extra_data=False):
        for col, row, cell in self.offline_cells():
            print(col, row, cell)
            for dir in range(4):
                for action in range(3):
                    print((col, row, dir))
                    # sets grid world in desired state and direction
                    self._gen_grid(self.width, self.height, (col, row), dir)
                    state_vector = torch.from_numpy(np.concatenate(self.construct_state_vector(self.agent_pos, self.agent_dir), axis=None)).float().unsqueeze(0)
                    # executes action, observes reward and next state
                    _, reward, _, _ = self.step(action)
                    new_cell = self.grid.get(self.agent_pos[0], self.agent_pos[1])
                    if new_cell is None or new_cell.type != 'lava':
                        next_state_trace = torch.from_numpy(np.concatenate(self.construct_state_vector(self.agent_pos, self.agent_dir), axis=None)).float().unsqueeze(0)
                        assert next_state_trace is not None, ('not none!', Transition(state=state_vector, action=action, reward=reward, next_state=next_state_trace))
                        reward = torch.tensor([[reward]], requires_grad=False)
                        action = torch.tensor([[action]], requires_grad=False)
                        print(Transition(state=state_vector, action=action, reward=reward, next_state=next_state_trace))
                        yield Transition(state=state_vector, action=action, reward=reward, next_state=next_state_trace)
                        if extra_data:
                            yield Transition(state=state_vector + torch.tensor([[0.15, 0, 0]], requires_grad=False), action=action, reward=reward, next_state=next_state_trace + torch.tensor([[0.15, 0, 0]], requires_grad=False))
                            yield Transition(state=state_vector + torch.tensor([[0, 0.15, 0]], requires_grad=False), action=action, reward=reward, next_state=next_state_trace + torch.tensor([[0, 0.15, 0]], requires_grad=False))