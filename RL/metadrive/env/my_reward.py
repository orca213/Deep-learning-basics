def my_reward_function(vehicle_speed, long_last, long_now):
    """
    TIP! original reward function was:
    reward = vehicle_speed / MAX_SPEED + road_progression
    """
    
    MAX_SPEED = 80 # km/h
    road_progression = long_now - long_last # meters per step
    
    # TODO: Implement a reward function
    # best reward when vehicle_speed / MAX_SPEED = 0.3 and road_progression = 0.3
    reward = 0
    reward += 1 - abs(vehicle_speed / MAX_SPEED - 0.3) / 0.3
    reward += 1 - abs(road_progression - 0.5) / 0.5
    
    return reward