from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Tuple
import numpy as np
import utils
import warnings
import subprocess

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

MAX_STEPS = 100
FREQUENCY_STEP = 0.5
PERIOD_STEP = 0.005

class SpectrogramEnv(Env):
    def __init__(self, file_name,initial_path, path=None):
        if path == None:
            path = "./"
        self.path = path
        self.state = utils.get_initial_state(path)
        #Action space (curve, freq, sign, period, phase_shift, integrand_sign)
          
        self.action_space = Tuple(( Discrete(28),           #Curve Index
                                    Discrete(3, start=-1),  #Frequency shift
                                    Discrete(2),            #Sign change
                                    Discrete(3, start=-1),  #Period Change
                                    Discrete(2),            #phase_shift change
                                    Discrete(2)))           #Integrand change
        self.curr_step = 0
        self.max_steps = MAX_STEPS
        self.freq_step = FREQUENCY_STEP
        self.p_step = PERIOD_STEP
        self.error_tol = 10

    def step(self, action):

        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        curve_index, shift_f, shift_sgn, shift_p, shift_ps, shift_integ = action

        f, sgn, p, ps, isgn = self.state[curve_index]

        # print(f"Action taken:\n \
        #       Curve Index:\t {curve_index}\n \
        #       Freq shift:\t {shift_f}\n \
        #       Sign change:\t {shift_sgn} \n \
        #       Period shift:\t {shift_p} \n \
        #       Isgn change:\t {shift_integ} \n \
        #       ")
        
        f += self.freq_step*shift_f
        sgn = sgn*1j.real if shift_sgn == 1 else sgn
        p += self.freq_step*shift_p
        ps = (ps+1)%2 if shift_ps == 1 else ps
        isgn *= -1 if shift_integ == 1 else ps

        self.state[curve_index] = [f, sgn, p, ps, isgn]
        e = 10000
        terminated = bool(e < self.error_tol
                          or self.curr_step > self.max_steps)
        if not terminated:
            reward = -1
        else:
            reward = 0
        
        info = {}
        self.curr_step+=1
        subprocess.call(["wolframscript", "-script", "spec_generator.m"])
        additional_reward = utils.compare_spectrograms("cur_spec.xlsx", "new_spec.xlsx")
        

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(self, seed=None):

        super().reset(seed=seed)
        
        self.curr_step = 0
        self.state = utils.get_initial_state(self.path)
        
        return np.array(self.state, dtype=np.float32)
        
def main():
    env = SpectrogramEnv(file_name="mathematica_spectrogram_2.xlsx", initial_path="./")
    episodes = 5

    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0

        while not done:
            action = env.action_space.sample()
            n_state, reward, done, trunc, info = env.step(action)
            
            score+= reward
        print(f"Episode: {episode} Score : {score}")

if __name__ == "__main__":
    main()



