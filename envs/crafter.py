import gym
import crafter
from utils.envs import OneHotAction


class CrafterEnv(gym.Env):
    
    def __init__(self):
        env = gym.make('CrafterReward-v1')
        env = crafter.Recorder(
          env, './logs',
          save_stats=False,
          save_video=False,
          save_episode=False)
        self.env = OneHotAction(env)
        c,h,w = self.reset()['image'].shape
        self.observation_space = gym.spaces.MultiBinary((c,h,w))
        
    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        image = self.env.reset()
        obs = {"image": image.transpose(2, 0, 1)}
        return obs

    def step(self, action):
        image, reward, done, info = self.env.step(action)
        # done = truncated or done
        obs = {"image": image.transpose(2, 0, 1)}
        return obs, reward, done, info

    def render(self, mode):
        return self.env.render(mode)
