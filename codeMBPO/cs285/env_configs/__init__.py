from .sac_config import sac_config
from .mpc_config import mpc_config
from .mpc_dos_config import mpc_dos_config
from .dqn_nanoslab import dqn_nanoslab_config

configs = {
    "sac": sac_config,
    "mpc": mpc_config,
    "dos": mpc_dos_config,
    "dqn": dqn_nanoslab_config,
}
