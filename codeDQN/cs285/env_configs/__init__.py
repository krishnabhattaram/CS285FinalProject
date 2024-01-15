from .dqn_atari_config import atari_dqn_config
from .dqn_basic_config import basic_dqn_config
from .dqn_nanoslab_config import nanoslab_dqn_config
from .sac_config import sac_config

configs = {
    "dqn_atari": atari_dqn_config,
    "dqn_basic": basic_dqn_config,
    "dqn": sac_config,
    "dqn_nanoslab": nanoslab_dqn_config,
}
