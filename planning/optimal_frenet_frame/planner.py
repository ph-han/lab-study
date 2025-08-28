import numpy as np
import matplotlib.pyplot as plt

def find_closest_waypoint(curr_x, curr_y, center_line_xlist, center_line_ylist):
    xlist = np.array(center_line_xlist)
    ylist = np.array(center_line_ylist)

    distance_list = np.hypot(xlist - curr_x, ylist - curr_y)
    closest_wp = np.argmin(distance_list)
    return closest_wp

def get_next_waypoint(curr_x, curr_y, center_line_xlist, center_line_ylist):
    closest_wp = find_closest_waypoint(curr_x, curr_y, center_line_xlist, center_line_ylist)

    # loop until the next waypoint is ahead
    while True: 
        traj_vec = np.array([center_line_xlist[closest_wp + 1] - center_line_xlist[closest_wp],
                            center_line_ylist[closest_wp + 1] - center_line_ylist[closest_wp]])
        ego_vec = np.array([curr_x - center_line_xlist[closest_wp], curr_y - center_line_ylist[closest_wp]])

        # check if waypoint is ahead of ego vehicle.
        is_waypoint_ahead = np.sign(np.dot(ego_vec, traj_vec)) 
        if is_waypoint_ahead < 0:
            break

        closest_wp += 1

    return closest_wp

def world2frenet(curr_x, curr_y, center_line_xlist, center_line_ylist):
    next_wp = get_next_waypoint(curr_x, curr_y, center_line_xlist, center_line_ylist)
    prev_wp = next_wp - 1


    ego_vec = np.array([
        curr_x - center_line_xlist[prev_wp], 
        curr_y - center_line_ylist[prev_wp]
    ])
    traj_vec = np.array([
        center_line_xlist[next_wp] - center_line_xlist[prev_wp],
        center_line_ylist[next_wp] - center_line_ylist[prev_wp]
    ])

    ego_proj_vec = ((ego_vec @ traj_vec) / (traj_vec @ traj_vec)) * traj_vec
    frenet_d_sign = np.sign(traj_vec[0] * ego_vec[1] - traj_vec[1] * ego_vec[0])    

    frenet_d = frenet_d_sign * np.hypot(ego_proj_vec[0] - ego_vec[0], ego_proj_vec[1] - ego_vec[1])

    frenet_s = 0
    for i in range(prev_wp):
        frenet_s += np.hypot(
            center_line_xlist[i + 1] - center_line_xlist[i],
            center_line_ylist[i + 1] - center_line_ylist[i]
        )
    
    return frenet_s, frenet_d

def frenet2world(curr_s, curr_d):
    pass

if __name__ == "__main__":
    center_line_xlist = np.linspace(10, 50, 40)
    center_line_ylist = 0.1 * (center_line_xlist**2)

    ego_x, ego_y = 25, 45
    frenet_s, frenet_d = world2frenet(ego_x, ego_y, center_line_xlist, center_line_ylist)

    print(f"frenet coordinate (s, d): ({frenet_s}, {frenet_d})")
    plt.plot(ego_x, ego_y, 'xb')
    plt.plot(center_line_xlist, center_line_ylist)
    plt.show()
