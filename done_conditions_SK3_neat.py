def done_conditions(lives_left, done,
                    x_pos_current, modify_limit_val,
                    fitness_incr_counter, fitness_incr_limit, 
                    steps_in_place, in_place_limit, round_expire_scale
                    ):

    ### If sonic loses a life, done.
    if lives_left < 3:
        print("\tLost a life")
        done = True

    ### If X position is greater than the modify limit, scale the counter limit
    if (x_pos_current > modify_limit_val):
        in_place_limit = round_expire_scale * in_place_limit
    ### If steps in place counter is higher than limit, done.
    if (steps_in_place == in_place_limit):
        print("\t>{} steps in place. // x value at end : {}".format(in_place_limit, x_pos_current))
        done = True


    ### If X position is greater than the modify limit, scale the counter limit
    if x_pos_current > modify_limit_val:
            fitness_incr_limit = round_expire_scale * fitness_incr_limit
    ### If fitness increasing counter is higher than limit, done.
    if (fitness_incr_counter == fitness_incr_limit):
        print("\tFitness hasn't increased in {} steps. // x value at end : {}".format(fitness_incr_limit, x_pos_current))
        done = True
        
    return done
    
    