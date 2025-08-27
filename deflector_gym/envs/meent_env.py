from functools import partial
import logging

import gym
import numpy as np

from JLAB.solver import JLABCode
from .base import DeflectorBase
from .actions import Action1D2, Action1D4

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def badcell(img, mfs):
    img = np.array(img)
    len = img.size
    sz = mfs+2
    window = np.ones(sz)
    window[0] = -1
    window[-1] = -1
    imgextend = np.concatenate((img, img))
    volved = np.convolve(imgextend, window)
    output = volved[sz-1:len+sz-1]

    return np.sum(np.floor(np.abs(output/sz)))

def underMFS(img, mfs):
    num = 0
    for i in range(1,mfs):
        num += badcell(img, i)
    return num

class MeentBase(DeflectorBase):
    def __init__(
            self,
            n_cells=256,
            wavelength=1100,
            desired_angle=70,
            order=40,
            thickness=325,
            refractive_index=1.45,
            *args,
            **kwargs,
    ):
        super().__init__(n_cells, wavelength, desired_angle, order, thickness, 
                         refractive_index)

    def get_efficiency(self, struct):
        # struct [1, -1, 1, 1, ...]
        struct = struct[np.newaxis, np.newaxis, :]

        wls = np.array([self.wavelength])
        period = abs(wls / np.sin(self.desired_angle / 180 * np.pi))
        calc = JLABCode(
            grating_type=0,
            n_I=self.refractive_index, n_II=1., theta=0, phi=0.,
            fourier_order=self.order, period=period,
            wls=wls, pol=1,
            patterns=None, ucell=struct, thickness=np.array([self.thickness])
        )

        eff, _, _ = calc.reproduce_acs_cell('p_si__real', 1)

        return eff


class MeentIndex(MeentBase):
    def __init__(
            self,
            n_cells=256,
            wavelength=1100,
            desired_angle=70,
            *args,
            **kwargs
    ):
        super().__init__(n_cells, wavelength, desired_angle, *args, **kwargs)

        self.observation_space = gym.spaces.Box(
            low=-1., high=1.,
            shape=(n_cells,), #### TODO fix shape
            dtype=np.float64
        )
        self.action_space = gym.spaces.Discrete(n_cells)

    def reset(self):
        self.struct = self.initialize_struct()
        self.eff = self.get_efficiency(self.struct)

        return self.struct.copy()

    def step(self, action):
        prev_eff = self.eff

        self.flip(action)
        self.eff = self.get_efficiency(self.struct)

        reward = self.eff - prev_eff

        # unsqueeze for 1 channel
        return self.struct.copy(), reward, False, {}


def initialize_agent(initial_pos, n_cells):
    # initialize agent
    if initial_pos == 'center':
        pos = n_cells // 2
    elif initial_pos == 'right_edge':
        pos = n_cells - 1
    elif initial_pos == 'left_edge':
        pos = 0
    elif initial_pos == 'random':
        pos = np.random.randint(n_cells)
    else:
        raise RuntimeError('Undefined inital position')

    return pos


class MeentAction1D2(MeentBase):
    def __init__(
            self,
            n_cells=256,
            wavelength=1100,
            desired_angle=70,
            refractive_index=1.45,
            initial_pos='center',  # initial agent's position
            *args,
            **kwargs
    ):
        super().__init__(n_cells, wavelength, desired_angle, refractive_index)

        self.observation_space = gym.spaces.Box(
            low=-1., high=1.,
            shape=(2*n_cells,),
            dtype=np.float64
        )
        self.action_space = gym.spaces.Discrete(len(Action1D2)) # start=-1
        self.initial_pos = initial_pos
        self.onehot = np.eye(n_cells)

    def reset(self):
        # initialize structure

        self.struct = self.initialize_struct()
        self.eff = self.get_efficiency(self.struct)
        self.pos = initialize_agent(self.initial_pos, self.n_cells)

        return np.concatenate((self.struct, self.onehot[self.pos]))

    def step(self, ac):
        prev_eff = self.eff
        # left == -1, noop == 0, right == 1
        # this way we can directly use ac as index difference
        ac -= 1

        self.flip(self.pos + ac)
        self.eff = self.get_efficiency(self.struct)

        reward = self.eff - prev_eff

        return np.concatenate((self.struct, self.onehot[self.pos])), reward, False, {}

class MeentAction1D4(MeentBase):
    def __init__(
            self,
            n_cells=256,
            wavelength=1100,
            desired_angle=70,
            refractive_index=1.45,
            initial_pos='center',  # initial agent's position
            *args,
            **kwargs
    ):
        super().__init__(n_cells, wavelength, desired_angle, refractive_index)

        self.observation_space = gym.spaces.Box(
            low=-1., high=1.,
            shape=(2*n_cells,),
            dtype=np.float64
        )
        self.action_space = gym.spaces.Discrete(len(Action1D4))
        self.initial_pos = initial_pos
        self.onehot = np.eye(n_cells)

    def reset(self):
        # initialize structure

        self.struct = self.initialize_struct(n_cells=self.n_cells)
        self.eff = self.get_efficiency(self.struct)
        self.pos = initialize_agent(self.initial_pos, self.n_cells)

        return np.concatenate((self.struct, self.onehot[self.pos]))

    def step(self, ac):
        prev_eff = self.eff

        if ac == Action1D4.RIGHT_SI.value and self.pos + 1 < self.n_cells:
            self.pos += 1
            self.struct[self.pos] = 1
        elif ac == Action1D4.RIGHT_AIR.value and self.pos + 1 < self.n_cells:
            self.pos += 1
            self.struct[self.pos] = -1
        elif ac == Action1D4.LEFT_SI.value and 0 <= self.pos - 1:
            self.pos -= 1
            self.struct[self.pos] = 1
        elif ac == Action1D4.LEFT_AIR.value and 0 <= self.pos - 1:
            self.pos -= 1
            self.struct[self.pos] = -1

        self.eff = self.get_efficiency(self.struct)

        reward = self.eff - prev_eff

        return np.concatenate((self.struct, self.onehot[self.pos])), reward, False, {}



class MultiRIIndex(MeentBase):
    def __init__(
        self,
        n_cells=256,
        wavelength=1100,
        desired_angle=70,
        order=40,
        thickness=325,
        refractive_index=1.45,
        refractive_index_2=1.0,
        reward_mode="shaped",
        lambda_off=1.0,
        gamma_delta=0.05,
    ):
        # explicit, refactor away from kwargs.get
        super().__init__(n_cells, wavelength, desired_angle, order, thickness, refractive_index)
        self.reward_mode = reward_mode
        self.lambda_off = lambda_off
        self.gamma_delta = gamma_delta

        # on/off refractive indices
        self.ri_on = refractive_index
        self.ri_off = refractive_index_2
        
        self.eff_on = 0.0
        self.eff_off = 0.0

        self.observation_space = gym.spaces.Box(
            low=-1., high=1.,
            shape=(self.n_cells,),
            dtype=np.float64
        )
        self.action_space = gym.spaces.Discrete(self.n_cells)
        
    def get_efficiency(self, struct):
        self.eff_on = float(self._compute(self.ri_on, struct))
        self.eff_off = float(self._compute(self.ri_off, struct))
        return self.eff_on - self.eff_off
        
    def reset(self):
        self.struct = self.initialize_struct()
        self.eff = self.get_efficiency(self.struct)
        return self.struct.copy()

    def step(self, action):
        prev_on, prev_off = self.eff_on, self.eff_off
        self.flip(action)
        self.eff = self.get_efficiency(self.struct)

        reward = self.calculate_reward(
            self.eff_on, self.eff_off, prev_on, prev_off
        )

        # log both as custom_metrics
        info = {'eff_on':  self.eff_on,
                'eff_off': self.eff_off}
        return self.struct.copy(), reward, False, info
    
    def calculate_reward(self, eff_on, eff_off, prev_eff_on, prev_eff_off):
        margin = eff_on - eff_off
        prev_margin = prev_eff_on - prev_eff_off
        if self.reward_mode == "margin":
            return margin
        if self.reward_mode == "weighted_margin":
            return eff_on - self.lambda_off * eff_off
        if self.reward_mode == "margin_delta":
            return margin + self.gamma_delta * (margin - prev_margin)
        if self.reward_mode == "ratio":
            return eff_on / (eff_off + 1e-6)
        if self.reward_mode == "log_ratio":
            return np.log((eff_on + 1e-6) / (eff_off + 1e-6))
        if self.reward_mode == "shaped":
            on_n  = np.clip(eff_on / 100.0, 0.0, 1.0)
            off_n = np.clip(eff_off / 100.0, 0.0, 1.0)
            base  = 1.0 - (1.0 - on_n)**2
            pen   = off_n**2
            delta = self.gamma_delta * ((eff_on - prev_eff_on) - (eff_off - prev_eff_off)) / 100.0
            return float(base - pen + delta)
        else:
            # original scheme: delta improvement in margin
            # (eff_on - prev_eff_on) - (eff_off - prev_eff_off) == margin - prev_margin
            return float((eff_on - prev_eff_on) - (eff_off - prev_eff_off))

    def _compute(self, ri, struct):
        # same core as MeentBase.get_efficiency but with n_I=ri
        struct = struct[np.newaxis, np.newaxis, :]

        wls = np.array([self.wavelength])
        period = abs(wls / np.sin(self.desired_angle / 180 * np.pi))
        calc = JLABCode(
            grating_type=0,
            n_I=ri, n_II=1., theta=0, phi=0.,
            fourier_order=self.order, period=period,
            wls=wls, pol=1,
            patterns=None, ucell=struct, thickness=np.array([self.thickness])
        )

        eff, _, _ = calc.reproduce_acs_cell('p_si__real', 1)

        return eff