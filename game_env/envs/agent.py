"""Base class for an agent that defines the possible actions. """
import sys
sys.path.append("..")
from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np
import utility_funcs as util
from collections import deque


# basic moves every agent should do
BASE_ACTIONS = {0: 'MOVE_LEFT',  # Move left
                1: 'MOVE_RIGHT',  # Move right
                2: 'MOVE_UP',  # Move up
                3: 'MOVE_DOWN',  # Move down
                4: 'STAY',  # don't move
                5: 'TURN_CLOCKWISE',  # Rotate counter clockwise
                6: 'TURN_COUNTERCLOCKWISE',  # Rotate clockwise
                7: 'FIRE'}


class Agent(object):

    def __init__(self, agent_id, start_pos, start_orientation, grid, row_size, col_size):
        """Superclass for all agents.

        Parameters
        ----------
        agent_id: (str)
            a unique id allowing the map to identify the agents
        start_pos: (np.ndarray)
            a 2d array indicating the x-y position of the agents
        start_orientation: (np.ndarray)
            a 2d array containing a unit vector indicating the agent direction
        grid: (2d array)
            a reference to this agent's view of the environment
        row_size: (int)
            how many rows up and down the agent can look
        col_size: (int)
            how many columns left and right the agent can look
        """
        self.agent_id = agent_id
        self.pos = np.array(start_pos)
        self.orientation = start_orientation
        # TODO(ev) change grid to env, this name is not very informative
        self.grid = grid
        self.row_size = row_size
        self.col_size = col_size
        self.reward_this_turn = 0
        self.tagged=False


        #IMRL


        self.core = 'wf' # fw or wf (fairness then wellbing, or wellbeing then fairness)
        self.wellbeing_fx = 'variance' # absolute, variance, aspiration

        #TASK game has to match rllib's and set alpha appropriately
        #elibility trace
        self.gamma = 0.99 #
        self.eligibility_trace = 0
        self.el_alpha=0.5
        
        #defection in M time
        self.context_memory = 1000
        self.defectingDeque = deque(maxlen=self.context_memory)
        

        
        
        self.R = 13 
        self.T = 5 #GameSetting.TAGGED_TIME - GameSetting.APPLE_RESPAWN_TIME/self.R
        self.S = 0
        self.P = 0 
        #A0
        self.aspirational = self.R + self.T + self.S + self.P
        self.aspiration_beta = 0.2 # aspiration learning rate

        #Core Derivation function
        self.f_u = 1 # conceder 0 < u < 1, linear: u=1, and boulware u>u
        #secondarary emotion derivation g_x
        self.g_v = 1 # 0.2 0.6 1, 2, 3, 5, 10 
    def reset(self):
        raise NotImplementedError

    def update_internal(self, action, _reward, neigbors, iter_time):
        
        self.update_eligibility(_reward)
        self.update_defection(action, iter_time)
        in_reward = self.emotional_derivation(_reward, neigbors)

        return in_reward

        

    
    def update_eligibility(self, _reward):
        self.eligibility_trace = self.gamma*self.el_alpha*self.eligibility_trace + _reward
    
    def update_defection(self, _action, _iter_time):
        # consider only defection in last M steps
        # decection detected
        if _action == 7:
            self.defectingDeque.append(_iter_time)
        # forgot a detection
        if self.defection_n > 0:
            if  self.defectingDeque[0]+self.context_memory < _iter_time:
                self.defectingDeque.popleft()
    @property
    def defection_n(self):
        return len(self.defectingDeque)

    def social_fairness_context(self, neightbors):
        #cn = 1/N sum((ni_c-ni_d/M)
        Cn=0
        for agent in neightbors:
            Cn += (agent.context_memory-2*agent.defection_n)/agent.context_memory
        Cn /=max(1,len(neightbors))

        return Cn;
        
    
    def social_fairness_appraisal(self, neightbors):
        context = self.social_fairness_context(neightbors)
        cooperating_rate = self.context_memory-2*self.defection_n
        F = context * (cooperating_rate)/self.context_memory #F = cn x (cn-nd)/M
        # print(context, cooperating_rate)
        
        return F, cooperating_rate < 0 and context >0,  cooperating_rate > 0 and context < 0

    def wellbeing_appraisal(self, _reward):
        #social dilemma payoff
        T = self.T #tempetation
        S = self.S #sucker
        W = 0
        try:
            if self.wellbeing_fx == 'absolute':
                # W = (2*self.eligibility_trace - self.context_memory*(T-S))/self.context_memory*(T-S) # w = (2r_t - Mx(T-S))/Mx(T-S)
                W = (2*self.eligibility_trace - self.context_memory*max(1, T-S))/(self.context_memory*max(1, T-S))
            elif self.wellbeing_fx  == 'variance':
                W = (self.gamma*self.el_alpha*self.eligibility_trace + _reward - self.eligibility_trace)/(self.context_memory*(T-S)) # w= (r_t+1 - r_t)/Mx(T-S)
            elif self.wellbeing_fx == 'aspiration':
                h = 1
                W = np.tanh(h*(self.eligibility_trace/(self.context_memory-self.aspirational)))
                self.aspirational = (1-self.aspiration_beta)*self.aspirational + self.aspiration_beta*(self.eligibility_trace/self.context_memory)
            else:
                print("the wellbeing function not known")
                import sys
                sys.exit(1)
        except Exception as err:
            print(err)
            import sys
            sys.exit(1)

        return W
                
    def emotional_derivation(self, _reward, neighbors):
        wellbeing_appraisal = self.wellbeing_appraisal(_reward)
        fairness_appraisal, exploiting, manipulated = self.social_fairness_appraisal(neighbors)


        E_joy = 0
        E_sad = 0
        E_anger = 0
        E_fearful = 0

        if self.core=='fw':
            print("Using FW")
            if fairness_appraisal>0:
                if wellbeing_appraisal>0:
                    E_joy = self.core_f(fairness_appraisal)*self.secondary_g(wellbeing_appraisal)
            elif fairness_appraisal<0:
                #agents either defect more in a cooperative environement
                if exploiting:
                    E_fearful = -(self.core_f(-1*fairness_appraisal)*self.secondary_g(wellbeing_appraisal))
                elif manipulated:
                    E_anger = -(self.core_f(-1*fairness_appraisal)*self.secondary_g(-1*wellbeing_appraisal))

        elif self.core == 'wf':
            print("Using WF")
            if wellbeing_appraisal >0:
                E_joy = self.core_f(wellbeing_appraisal)*self.secondary_g(fairness_appraisal)
            elif wellbeing_appraisal <0:
                E_sad = -(self.core_f(-1*wellbeing_appraisal)*self.core_f(fairness_appraisal))
        emotions = [E_joy>0, E_sad<0, E_anger<0, E_fearful<0]
        print(emotions)
        assert(sum(emotions) < 2, "detected more than one emotions")

        return E_joy+E_sad+E_fearful+E_anger

    # Core Derivation Function monotonically maps the desireability of emotion to [0, 1]
    def core_f(self, D_x):
        return D_x**self.f_u

    
    # secondary emotional derivation that maps emotional Intensity [-1, 1] to value [0-1]
    def secondary_g(self, I_x):
        return ((I_x + 1)/2)**self.g_v

    @property
    def action_space(self):
        """Identify the dimensions and bounds of the action space.

        MUST BE implemented in new environments.

        Returns
        -------
        gym Box, Discrete, or Tuple type
            a bounded box depicting the shape and bounds of the action space
        """
        raise NotImplementedError

    @property
    def observation_space(self):
        """Identify the dimensions and bounds of the observation space.

        MUST BE implemented in new environments.

        Returns
        -------
        gym Box, Discrete or Tuple type
            a bounded box depicting the shape and bounds of the observation
            space
        """
        raise NotImplementedError

    def action_map(self, action_number):
        """Maps action_number to a desired action in the map"""
        raise NotImplementedError

    def get_state(self):
        return util.return_view(self.grid, self.get_pos(),
                                self.row_size, self.col_size)

    def compute_reward(self):
        reward = self.reward_this_turn
        self.reward_this_turn = 0
        return reward
    def istagged(self):
        temp = self.tagged
        self.tagged = False #reset for the next step
        return temp

    def set_pos(self, new_pos):
        self.pos = np.array(new_pos)

    def get_pos(self):
        return self.pos

    def translate_pos_to_egocentric_coord(self, pos):
        offset_pos = pos - self.get_pos()
        ego_centre = [self.row_size, self.col_size]
        return ego_centre + offset_pos

    def set_orientation(self, new_orientation):
        self.orientation = new_orientation

    def get_orientation(self):
        return self.orientation

    def get_map(self):
        return self.grid

    def return_valid_pos(self, new_pos):
        """Checks that the next pos is legal, if not return current pos"""
        ego_new_pos = new_pos  # self.translate_pos_to_egocentric_coord(new_pos)
        new_row, new_col = ego_new_pos
        # you can't walk through walls
        temp_pos = new_pos.copy()
        if self.grid[new_row, new_col] == '@':
            temp_pos = self.get_pos()
        return temp_pos

    def update_agent_pos(self, new_pos):
        """Updates the agents internal positions

        Returns
        -------
        old_pos: (np.ndarray)
            2 element array describing where the agent used to be
        new_pos: (np.ndarray)
            2 element array describing the agent positions
        """
        old_pos = self.get_pos()
        ego_new_pos = new_pos  # self.translate_pos_to_egocentric_coord(new_pos)
        new_row, new_col = ego_new_pos
        # you can't walk through walls
        temp_pos = new_pos.copy()
        if self.grid[new_row, new_col] == '@':
            temp_pos = self.get_pos()
        self.set_pos(temp_pos)
        # TODO(ev) list array consistency
        return self.get_pos(), np.array(old_pos)

    def update_agent_rot(self, new_rot):
        self.set_orientation(new_rot)

    def hit(self, char):
        """Defines how an agent responds to being hit by a beam of type char"""
        raise NotImplementedError

    def consume(self, char):
        """Defines how an agent interacts with the char it is standing on"""
        raise NotImplementedError


HARVEST_ACTIONS = BASE_ACTIONS.copy()
HARVEST_ACTIONS.update({7: 'FIRE'})  # Fire a penalty beam

HARVEST_VIEW_SIZE = 7


class HarvestAgent(Agent):

    def __init__(self, agent_id, start_pos, start_orientation, grid, view_len=HARVEST_VIEW_SIZE):
        self.view_len = view_len
        super().__init__(agent_id, start_pos, start_orientation, grid, view_len, view_len)
        self.update_agent_pos(start_pos)
        self.update_agent_rot(start_orientation)
    def reset(self):
        self.defectingDeque.clear()
        self.eligibility_trace= 0

    @property
    def action_space(self):
        return Discrete(8)

    # Ugh, this is gross, this leads to the actions basically being
    # defined in two places
    def action_map(self, action_number):
        """Maps action_number to a desired action in the map"""
        return HARVEST_ACTIONS[action_number]

    @property
    def observation_space(self):
        return Box(low=0, high=255, shape=(2 * self.view_len + 1,
                                             2 * self.view_len + 1, 3), dtype=np.uint8)

    def hit(self, char):
        if char == 'F':
            self.tagged = True
            self.reward_this_turn -= 50

    def fire_beam(self, char):
        if char == 'F':
            self.reward_this_turn -= 1

    def get_done(self):
        return False

    def consume(self, char):
        """Defines how an agent interacts with the char it is standing on"""
        if char == 'A':
            self.reward_this_turn += 1
            return ' '
        else:
            return char


CLEANUP_ACTIONS = BASE_ACTIONS.copy()
CLEANUP_ACTIONS.update({7: 'FIRE',  # Fire a penalty beam
                        8: 'CLEAN'})  # Fire a cleaning beam

CLEANUP_VIEW_SIZE = 7


class CleanupAgent(Agent):
    def __init__(self, agent_id, start_pos, start_orientation, grid, view_len=CLEANUP_VIEW_SIZE):
        self.view_len = view_len
        super().__init__(agent_id, start_pos, start_orientation, grid, view_len, view_len)
        # remember what you've stepped on
        self.update_agent_pos(start_pos)
        self.update_agent_rot(start_orientation)
    def reset(self):
        self.defectingDeque.clear()
        self.eligibility_trace= 0

    @property
    def action_space(self):
        return Discrete(9)

    @property
    def observation_space(self):
        return Box(low=0, high=255, shape=(2 * self.view_len + 1,
                                             2 * self.view_len + 1, 3), dtype=np.uint8)

    # Ugh, this is gross, this leads to the actions basically being
    # defined in two places
    def action_map(self, action_number):
        """Maps action_number to a desired action in the map"""
        return CLEANUP_ACTIONS[action_number]

    def fire_beam(self, char):
        if char == 'F':
            self.reward_this_turn -= 1

    def get_done(self):
        return False

    def hit(self, char):
        if char == 'F':
            self.tagged = True
            self.reward_this_turn -= 50

    def consume(self, char):
        
        """Defines how an agent interacts with the char it is standing on"""
        if char == 'A':
            self.reward_this_turn += 1
            return ' '
        else:
            return char
