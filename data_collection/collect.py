import copy
import numpy as np
from tqdm import tqdm
from envs import init_envs

from cli.collect import CollectArgs
from data_collection.agents import get_agent
from data_collection.configs.collect_configs import get_collect_config
from data_collection.constants import Environments
from data_collection.utils import get_environment, init_lib_seed, construct_blacklist, save_state_ids, save_actions, \
    delete_episode_observations, save_obs, check_duplication


init_envs()

def collect(args: CollectArgs):
    env = get_environment(args.environment, args.seed)
    init_lib_seed(args.seed)
    
    collect_config = get_collect_config(args.environment, args.split)
    agent = get_agent(env, args.environment, collect_config, args.device)
    
    shapes_env = args.environment in [Environments.CUBES_3D, Environments.SHAPES_2D]
    atari_env = args.environment != Environments.CRAFTER and not shapes_env
    
    blacklist = construct_blacklist(collect_config.blacklist_paths, atari_env)

    ep_idx = 0
    shapes_env = args.environment in [Environments.CUBES_3D, Environments.SHAPES_2D]
    atari_env = args.environment != Environments.CRAFTER and not shapes_env
    with tqdm(total=args.episodes) as pbar:
        
        while ep_idx < args.episodes:
            burnin_steps = 0
            if collect_config.max_burnin and collect_config.min_burnin < 0:
                
                burnin_steps = np.random.randint(collect_config.min_burnin,
                                                 collect_config.max_burnin)
                
            episode_states = []
            episode_actions = []
            agent.reset()
            prev_obs = env.reset()[0]
            step_idx = 0
            for _ in range(burnin_steps):
                action = agent.get_action(prev_obs, True)
                prev_obs, *_ = env.step(action)
            after_warmup = True

            while True:

                # TODO Refactor this

                if after_warmup:
                    action = 0
                else:
                    action = agent.get_action(prev_obs)

                step_state = env.step(action)
                if len(step_state) == 5:
                    obs, _, truncated, terminated, info = step_state
                    done = terminated
                else:
                    obs, _, done, info = step_state



                state = None

                if atari_env:
                    state = copy.deepcopy(
                        np.array(env.ale.getRAM(), dtype=np.int32))
                    
                if shapes_env:
                    state = info['state']

                if after_warmup:
                    after_warmup = False
                    if (atari_env or shapes_env) and check_duplication(blacklist, state):
                        break

                # if collect_config.crop:
                #     obs = crop_normalize(obs, collect_config.crop)
                episode_actions.append(action)
                step_idx += 1
                save_obs(
                    ep_idx,
                    step_idx,
                    obs,
                    collect_config.save_path,
                    atari_env or shapes_env,
                )
                if state is not None:
                    episode_states.append(state)

                if step_idx >= args.steps:
                    done = True

                if done:
                    if step_idx < args.steps:
                        delete_episode_observations(collect_config.save_path, ep_idx)
                        break
                    save_actions(episode_actions, ep_idx, collect_config.save_path)
                    save_state_ids(episode_states, ep_idx, collect_config.save_path)
                    ep_idx += 1
                    pbar.update(n=1)

                    break
