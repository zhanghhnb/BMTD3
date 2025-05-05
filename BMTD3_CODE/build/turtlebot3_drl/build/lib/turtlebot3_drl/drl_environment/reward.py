from ..common.settings import REWARD_FUNCTION, COLLISION_OBSTACLE, COLLISION_WALL, TUMBLE, SUCCESS, TIMEOUT, RESULTS_NUM

goal_dist_initial = 0

reward_function_internal = None

def get_reward(succeed, action_linear, action_angular, distance_to_goal, goal_angle, min_obstacle_distance):
    return reward_function_internal(succeed, action_linear, action_angular, distance_to_goal, goal_angle, min_obstacle_distance)

def get_reward_A(succeed, action_linear, action_angular, goal_dist, goal_angle, min_obstacle_dist):
        # [-3.14, 0]
        r_yaw = -2 * abs(goal_angle)  

        # [-4, 0]
        r_vangular = -1 * (action_angular**2)

        # [-1, 1]
        r_distance = ((2 * goal_dist_initial) / (goal_dist_initial + goal_dist) - 1)

        # [-20, 0]
        if min_obstacle_dist < 0.22:
            r_obstacle = -20
        else:
            r_obstacle = 0

        # [-2 * (2.2^2), 0]
        r_vlinear = -1 * (((0.22 - action_linear) * 10) ** 2)

        reward = r_yaw + r_distance + r_obstacle + r_vlinear + r_vangular - 1

        if succeed == SUCCESS:
            reward += 10000
        elif succeed == COLLISION_OBSTACLE or succeed == COLLISION_WALL:
            reward -= 2000
        return float(reward)

def get_reward_B(succeed, action_linear, action_angular, goal_dist, goal_angle, min_obstacle_dist):

    r_yaw = -1 * abs(goal_angle)

    r_vangular = -1 * (action_angular ** 2)

    r_distance = 2 * ((2 * goal_dist_initial) / (goal_dist_initial + goal_dist) - 1)

    if min_obstacle_dist < 0.5:  
        r_obstacle = -20 * (0.5 - min_obstacle_dist)
    elif min_obstacle_dist < 1.0:  
        r_obstacle = -10 * (1.0 - min_obstacle_dist)
    else: 
        r_obstacle = 10 * (min_obstacle_dist - 1.0)


    r_vlinear = -1 * (((0.22 - action_linear) * 10) ** 2)


    if min_obstacle_dist < 0.8 and action_linear > 0.2:
        r_vlinear -= 10  


    r_goal_reach = 0
    if goal_dist < 0.45:
        r_goal_reach = 100 


    reward = r_yaw + r_distance + r_obstacle + r_vlinear + r_vangular + r_goal_reach - 1


    if succeed == SUCCESS:
        reward += 5000
    elif succeed == COLLISION_OBSTACLE or succeed == COLLISION_WALL:
        reward -= 2000

    return float(reward)



def reward_initalize(init_distance_to_goal):
    global goal_dist_initial
    goal_dist_initial = init_distance_to_goal

function_name = "get_reward_" + REWARD_FUNCTION
reward_function_internal = globals()[function_name]
if reward_function_internal == None:
    quit(f"Error: reward function {function_name} does not exist")
