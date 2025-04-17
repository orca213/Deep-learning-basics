from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
import logging

CONFIG = {
    "traffic_density": 0,
    "accident_prob": 0,
    "crash_vehicle_done": True,
    "crash_object_done": True,
    "horizon": 1000,
    "use_lateral_reward": False,
    "random_spawn_lane_index": False,
    "use_render": False,
    "num_scenarios": 1,
    "start_seed": 1043,
    "log_level": logging.CRITICAL,
    "map_config": {
        BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
        BaseMap.GENERATE_CONFIG: 3,  # it can be a file path / block num / block ID sequence
        BaseMap.LANE_WIDTH: 4,
        BaseMap.LANE_NUM: 4,
        "exit_length": 50,
        "start_position": [0, 0],
    },
}