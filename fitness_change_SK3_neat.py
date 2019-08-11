def adjust_fitness(fitness_current, done, frame, 
                    x_pos_last, x_pos_current, 
                    current_max_x_pos, screen_x_end, x_pos_flags_ratios, 
                    x_pos_flags_points, 
                    points_flag=False, ratio_flag=True):
                    
    ### Remove 15 fitness every 100 frames
    if frame % 100 == 0:
        fitness_current -= 15
    ### Remove 50 fitness every 500 frames
    if frame % 500 == 0:
        fitness_current -= 50
    ### Remove 0.02 fitness per frame that the change in X is < 2
    if abs(x_pos_last - x_pos_current) < 2:
        fitness_current -= 0.02
    ### Add lots of points if agent makes it to the end of the level
    if x_pos_current == screen_x_end :
        fitness_current += 300000
        done = True
                
    ### Add 1 fitness for a new max distance
    if x_pos_current > current_max_x_pos:
        #fitness_current += min(abs(x_pos_current - current_max_x_pos), 2)
        fitness_current += 1
    
    """
    The following two if statements are to reward fitness for specific distance milestones.
    The first block, labeled with "if points_flag == True:" will reward a fixed number of points for a milestone
    """
    if points_flag == True:
        for key in x_pos_flags_points.keys():
            if (x_pos_current > key) and (x_pos_flags_points[key][0] == 0):
                if frame > 2:
                    fitness_current += x_pos_flags_points[key][1]
                    print("\tx == {} achieved! {} reward given.".format(key, x_pos_flags_points[key][1]))
                else:
                    print("\tx == {} achieved suspiciously quickly, no reward given.".format(key, x_pos_flags_points[key][1]))
                x_pos_flags_points[key][0] = 1
    
    """
    This block will award a multiplier for reach distance milestones. 
    This should work better in theory but is difficult due to negative fitness values.
    """    
    if ratio_flag == True:
        for key in x_pos_flags_ratios.keys():
            if (x_pos_current > key) and (x_pos_flags_ratios[key][0] == 0):
                if frame > 2:
                    fitness_current = fitness_current * x_pos_flags_ratios[key][1]
                    print("\tx == {} achieved! {} reward given.".format(key, x_pos_flags_ratios[key][1]))
                else:
                    print("\tx == {} achieved suspiciously quickly, no reward given.".format(key, x_pos_flags_ratios[key][1]))
                x_pos_flags_ratios[key][0] = 1
                
    
    
    return fitness_current, done, x_pos_flags_points, x_pos_flags_ratios