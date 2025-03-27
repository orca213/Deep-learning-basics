def my_reward_function(vehicle_speed, long_last, long_now):
    """
    TIP! original reward function was:
    reward = vehicle_speed / MAX_SPEED + road_progression
    """
    
    MAX_SPEED = 80 # km/h
    road_progression = long_now - long_last # meters per step
    
    # TODO: Implement a reward function
    reward = 0
    
    return reward