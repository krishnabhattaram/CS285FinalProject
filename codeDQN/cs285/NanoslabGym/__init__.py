import sys
sys.path.append('/Users/krishnabhattaram/Desktop/CS285/Project')

from codeMBPO.cs285.envs.NanoslabGym import NanoslabEnv
from codeMBPO.cs285.envs.NanoslabGym import NanoslabCachedEnv

from gym.envs.registration import register

register(
    id="NanoslabEnv-v0",
    entry_point="codeMBPO.cs285.envs.NanoslabGym.NanoslabEnv:NanoslabEnv",
    max_episode_steps=300,
)

register(
    id="NanoslabCachedEnv-v0",
    entry_point="codeMBPO.cs285.envs.NanoslabGym.NanoslabCachedEnv:NanoslabCachedEnv",
    max_episode_steps=300,
)

register(
    id="NanoslabSiCachedEnvSmall-v0",
    entry_point="codeMBPO.cs285.envs.NanoslabGym.NanoslabSiCachedEnvSmall:NanoslabSiCachedEnvSmall",
    max_episode_steps=300,
)

register(
    id="NanoslabSiCachedEnvLarge-v0",
    entry_point="codeMBPO.cs285.envs.NanoslabGym.NanoslabSiCachedEnvLarge:NanoslabSiCachedEnvLarge",
    max_episode_steps=300,
)
