from hal.constants import SHOULDER_CLUSTER_CENTERS_V0
from hal.constants import STICK_XY_CLUSTER_CENTERS_V0
from hal.constants import STICK_XY_CLUSTER_CENTERS_V1

TARGETS_EMBEDDING_SIZES = {
    "targets_v0": {
        "main_stick": len(STICK_XY_CLUSTER_CENTERS_V0),
        "c_stick": len(STICK_XY_CLUSTER_CENTERS_V0),
        "buttons": 6,
    },
    "targets_v1": {
        "main_stick": len(STICK_XY_CLUSTER_CENTERS_V1),
        "c_stick": len(STICK_XY_CLUSTER_CENTERS_V1),
        "shoulder": len(SHOULDER_CLUSTER_CENTERS_V0),
        "buttons": 6,
    },
}
