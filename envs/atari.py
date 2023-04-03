import gym
import gym.wrappers

from utils import OneHotAction


class Atari(gym.Env):

    def __init__(self,
                 name,
                 frame_skip=4,
                 size=64,
                 grayscale=False,
                 terminal_on_life_loss=False,
                 sticky_actions=False,
                 noops=30):

        env = gym.make(
            name, 
            # obs_type='rgb',
            obs_type='image',
            # render_mode='rgb_array',
            frameskip=frame_skip,
            full_action_space=False
        )
        env = gym.wrappers.AtariPreprocessing(
            env, 
            noop_max=noops, 
            frame_skip=1,
            screen_size=size,
            terminal_on_life_loss=terminal_on_life_loss, 
            grayscale_obs=grayscale
        )
        self.env = OneHotAction(env)
        self.grayscale = grayscale
        c,h,w = self.reset()['image'].shape
        self.observation_space = gym.spaces.MultiBinary((c,h,w))

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        # image, info = self.env.reset()
        image = self.env.reset()
        if self.grayscale:
            image = image[..., None]
        obs = {"image": image.transpose(2, 0, 1)}
        return obs

    def step(self, action):
        # image, reward, done, truncated, info = self.env.step(action)
        # done = truncated or done
        image, reward, done, info = self.env.step(action)
        if self.grayscale:
            image = image[..., None]
        obs = {"image": image.transpose(2, 0, 1)}
        return obs, reward, done, info

    def render(self, mode):
        return self.env.render(mode)
        