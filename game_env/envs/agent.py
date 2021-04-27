"""Base class for an agent that defines the possible actions. """
from os import system
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

        self.core = 'fw' # fw or wf (fairness then wellbing, or wellbeing then fairness)
        self.wellbeing_fx = 'variance' # variance, aspiration

        self.episode_len = 1000

        #elibility trace
        self.fairness_gamma = .99
        self.fairness_alpha = 1
        self.fairness_epsilon = 0.1

        self.prosocial_metric = 0

        self.reward_gamma = .99
        self.reward_alpha=1

        self.smoothen_wellbeing = 0
        


        self.aspirational = 0.5 
        self.aspiration_beta = 0.1 # aspiration learning rate
        

        #Core Derivation function
        self.f_u = 1 # conceder 0 < u < 1, linear: u=1, and boulware u>u
        #secondarary emotion derivation g_x
        self.g_v = 1 # 0.2 0.6 1, 2, 3, 5, 10 


         
    def reset(self):
        raise NotImplementedError

    def update_internal(self, _reward, neigbors, current_iter, is_cleanup):
        
        in_reward = self.emotional_derivation(_reward, neigbors, current_iter, is_cleanup)

        # update wellbeing
        self.update_wellbeing(_reward)
        return in_reward

        

    
    def update_wellbeing(self, _reward):
        self.smoothen_wellbeing = self.reward_gamma*self.reward_alpha*self.smoothen_wellbeing + _reward

    def update_prosocial(self, pro_s):
        self.prosocial_metric = self.fairness_gamma*self.fairness_alpha*self.prosocial_metric  +  pro_s

    def social_fairness_appraisal(self, neightbors, is_cleanup=False):
        if len(neightbors) == 0: return 0
        Cn=0
        temp_sum = self.prosocial_metric 
        reverse = 1
        if is_cleanup:
            reverse = -1
        for agent in neightbors:
            temp_sum += agent.prosocial_metric
            Cn += reverse*agent.prosocial_metric - reverse*self.prosocial_metric 
        return 0 if temp_sum == 0 else Cn/(temp_sum*len(neightbors) )

    def wellbeing_appraisal(self, _reward, current_iter):
        #social dilemma payoff
        # T = self.T #tempetation
        # S = self.S #sucker
        
        W = 0
        if self.wellbeing_fx  == 'variance':
            W = ((self.reward_gamma*self.reward_alpha*self.smoothen_wellbeing + _reward) - self.smoothen_wellbeing)/((current_iter-self.reward_gamma*current_iter)+51)
            # if _reward >=-1:
            #     print('A')
            #     W = ((self.reward_gamma*self.reward_alpha*self.smoothen_wellbeing + _reward) - self.smoothen_wellbeing)/(10*50+1) #current_iter- current_iter*self.reward_gamma)
            # else:
            #     W = ((self.reward_gamma*self.reward_alpha*self.smoothen_wellbeing + _reward) - self.smoothen_wellbeing)/(60+1)#(current_iter- current_iter*self.reward_gamma))
                
        elif self.wellbeing_fx == 'aspiration':
            h = 10
            W = np.tanh(h*((self.smoothen_wellbeing/current_iter) - self.aspirational))
            self.aspirational = (1-self.aspiration_beta)*self.aspirational + self.aspiration_beta*(self.smoothen_wellbeing/(current_iter+1))
        

        return W
                
    def emotional_derivation(self, _reward, neighbors, current_iter, is_cleaup):
        wellbeing_appraisal = self.wellbeing_appraisal(_reward, current_iter)
        fairness_appraisal = self.social_fairness_appraisal(neighbors, is_cleaup)

        

        #print("wellbeing: ",wellbeing_appraisal)
        #print("fariness appraisal: ",fairness_appraisal, len(neighbors))
        assert(abs(wellbeing_appraisal) <=1)
        assert(abs(fairness_appraisal) <=1)

        if len(neighbors) == 0:
            return wellbeing_appraisal, False, False, False, False

        E_joy = 0
        E_sad = 0
        E_anger = 0
        E_fearful = 0

        

        if self.core=='fw':
            # print("Using FW")
            if np.abs(fairness_appraisal) <=self.fairness_epsilon:
                # if wellbeing_appraisal>0:
                F = (self.fairness_epsilon-np.abs(fairness_appraisal))/self.fairness_epsilon
                E_joy = self.core_f(F) * self.secondary_g(wellbeing_appraisal)
            elif fairness_appraisal>0: #exploiting
                E_fearful = -self.core_f(abs(fairness_appraisal))*self.secondary_g(wellbeing_appraisal)
            else: ## same lines but be useful for stats
                E_anger = -self.core_f(abs(fairness_appraisal))*self.secondary_g(-1*wellbeing_appraisal)

        elif self.core == 'wf':
            # print("Using WF")
            
            if np.abs(fairness_appraisal) <=self.fairness_epsilon:
                F = (self.fairness_epsilon-np.abs(fairness_appraisal))/self.fairness_epsilon
            else:
                F =  -1.*abs(fairness_appraisal)
            
            if wellbeing_appraisal >0:
                E_joy = self.core_f(wellbeing_appraisal)*self.secondary_g(F)
            elif wellbeing_appraisal <0:
                E_sad = -(self.core_f(-1*wellbeing_appraisal)*self.secondary_g(F))
            else: # semi-egoists
                return F
        else:
            print("unknown emo function")
            sys.exit()
        emotions = [E_joy>0, E_sad<0, E_anger<0, E_fearful<0]
        # print(emotions)
        assert sum(emotions) <2, "detected more than one emotions"

        # print(f"joy {E_joy} + sad {E_sad} + fear {E_fearful} + anger {E_anger}")
        assert(E_fearful <= 0 and E_anger <=0 and E_sad<=0 and E_joy >=0)



        return E_joy + E_sad + E_fearful + E_anger, E_joy>0, E_sad<0, E_fearful<0, E_anger<0

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
        self.smoothen_wellbeing= 0
        self.prosocial_metric = 0

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
            self.reward_this_turn = -50

    def fire_beam(self, char):
        if char == 'F':
            self.reward_this_turn = -1

    def get_done(self):
        return False

    def consume(self, char):
        """Defines how an agent interacts with the char it is standing on"""
        if char == 'A':
            self.reward_this_turn += 1
            
            num_apple_in_area = self.count_apples(util.return_view(self.grid, self.get_pos(),
                                    2, 2))
            self.update_prosocial(num_apple_in_area**3)


            return ' '
        else:
            self.update_prosocial(0)
            return char
    def count_apples(self, window):
        # compute how many apples are in window
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_apples = counts_dict.get('A', 0)
        return num_apples


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
        self.smoothen_wellbeing= 0
        self.prosocial_metric = 0

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
            self.reward_this_turn = -1

    def get_done(self):
        return False

    def hit(self, char):
        if char == 'F':
            self.tagged = True
            self.reward_this_turn = -50

    def consume(self, char):
        
        """Defines how an agent interacts with the char it is standing on"""
        if char == 'A':
            self.reward_this_turn += 1
            return ' '
        else:
            return char
