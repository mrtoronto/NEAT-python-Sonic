import os
import retro
import numpy as np
import cv2
import neat
import pickle
from done_conditions_SK3_neat import done_conditions
from fitness_change_SK3_neat import adjust_fitness

env = retro.make('SonicAndKnuckles3-Genesis', 'AngelIslandZone.Act1')

### Set resume == 'True' to resume from checkpoint
resume = False
### Checkpoint files if checkpoint is to be loaded
start_checkpoint_file = "checkpoints/SK3_AIA1_250g_checkpoint-8"

### Prefix for checkpoints to be saved during run
checkpoint_prefix = "checkpoints/SK3_AIA1_250g_checkpoint-"

imgarray = []

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        observ = env.reset()
        ### Random action every time
        action = env.action_space.sample()
        
        ### Pixel values for the frame
        inx, iny, inc = env.observation_space.shape
        
        ### Reduce size of picture to speed up computation
        inx = int(inx/8)
        iny = int(iny/8)
        
        ### Run an RNN using the current genomes values
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        
        ### Set a bunch of values before the run
        ##### Max fitness value to be updated
        current_max_fitness = 0
        ##### Value that'll be used to hold the current fitness
        fitness_current = 0
        frame = 0
        ##### Counter that'll increment if the fitness doesn't increase
        fitness_incr_counter = 0
        ##### Values to hold X coordinates
        x_pos_current = 0
        x_pos_last = 0
        current_max_x_pos = 0
        ##### Counter for number of frames without changing X value
        steps_in_place = 0
        screen_x_end = 0

        #### Flags for whether the agent has hit a certain distance threshold
        flag_x_200 = 0
        flag_x_500 = 0
        flag_x_1000 = 0
        flag_x_2000 = 0
        flag_x_3000 = 0
        flag_x_4000 = 0
        flag_x_5000 = 0
        flag_x_6000 = 0
        flag_x_7000 = 0
        flag_x_8000 = 0
        flag_x_9000 = 0
        flag_x_10000 = 0
        
        ##### Dictionaries with milestone rewards
        ##### First option is a flat reward value per milestone
        x_pos_flags_points = {200 : [flag_x_200, 50],
                    1000 : [flag_x_1000, 1000],
                    2000 : [flag_x_2000, 1000],
                    3000 : [flag_x_3000, 1000],
                    4000 : [flag_x_4000, 1000],
                    5000 : [flag_x_5000, 1000],
                    6000 : [flag_x_6000, 1000],
                    7000 : [flag_x_7000, 1000],}
        ##### Second option is a multiplier per milestone
        ##### Be AWARE : If negative values, will make reward MORE negative
        x_pos_flags_ratios = {200 : [flag_x_200, 1.2],
                    1000 : [flag_x_1000, 1.2],
                    2000 : [flag_x_2000, 1.2],
                    3000 : [flag_x_3000, 1.2],
                    4000 : [flag_x_4000, 1.2],
                    5000 : [flag_x_5000, 1.2],
                    6000 : [flag_x_6000, 1.2],
                    7000 : [flag_x_7000, 1.2],
                    8000 : [flag_x_8000, 1.2],
                    9000 : [flag_x_9000, 1.2],
                    10000 : [flag_x_10000, 1.2],}
        
        
        
        done = False
        
        while not done:

            env.render()
            frame += 1
            ### Convert the current observational frame to greyscale
            observ = cv2.cvtColor(observ, cv2.COLOR_BGR2GRAY)
            
            ### Uncomment this to watch the processed frame while it plays
            #scaledimg = cv2.resize(observ, (iny, inx))
            #cv2.imshow('main', scaledimg)
            #cv2.waitKey(1)
            
            ### Resize and reshape the image array
            observ = cv2.resize(observ, (inx, iny))
            observ = np.reshape(observ, (inx,iny))
            imgarray = np.ndarray.flatten(observ)
            
            ### Use the NN defined earlier to predict the next movement
            nnOutput = net.activate(imgarray)

            ### Grab the max value from the nnOutput
            max_index = np.argmax([nnOutput])
            """
            These are the button options on the genesis
            Underneath are the 8 button combinations that are valuable in terms of progression
            Forcing the NN to predict one specific combination of buttons rather than a random group of them reduces the options it'll try from ~4096 to 8 which should speed up training.
            "buttons": 
            ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
            "relevant combinations":
            ( {{}, {LEFT}, {RIGHT}, {LEFT, DOWN}, {RIGHT, DOWN}, {DOWN}, {DOWN, B}, {B}} ).
            """
            if max_index == 0:
                # Nothing
                mod_step_control = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif max_index == 1:
                # Left
                mod_step_control = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
            elif max_index == 2:
                # Right
                mod_step_control = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            elif max_index == 3:
                # Down left
                mod_step_control = [0, 0, 0, 0, 0, 1, 1, 0, 0,0, 0, 0]
            elif max_index == 4:
                # Right down
                mod_step_control = [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
            elif max_index == 5:
                # Down
                mod_step_control = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            elif max_index == 6:
                # Down B
                mod_step_control = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            elif max_index == 7:
                # B
                mod_step_control = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            else:
                kprint("max index out of range. No action.")
                mod_step_control = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            
            """
            Below is the info dictionary collected with each frame.
            {'x': 313, 'lives': 3, 'level_end_bonus': 0, 'rings': 0, 'score': 0, 'zone': 7, 'act': 0, 'screen_x_end': 17048, 'screen_y': 1298, 'y': 1394, 'screen_x': 192, 'special_stage': 0}
            """
            ### Input the controls output by the NN to the next frame of the game
            observ, rew, done, info = env.step(mod_step_control)
            
            ### Every 100th frame, print the frame number and the current fitness
            if frame % 100 == 0:
                print("\tFrame : ", str(frame), " // fitness ", str(fitness_current))

            ### Check if sonic moved this step
            ### If previous X is less than current X, sonic just moved RIGHT and so reset the counter
            if x_pos_current < info['x']:
                steps_in_place = 0
                ### Don't actively penalize backward movement to encourage exploration
            else:
                ### Add to counter if he doesn't move right    
                steps_in_place += 1

            ### Capture data from info dictionary
            rings = info['rings']
            x_pos_current = info['x']
            screen_x_end = info['screen_x_end']
            screen_y = info['screen_y']
            y_pos_current = info['y']
            lives_left = info['lives']
            
            ### Run Fitness adjustment function
            ### Will output updated fitness, done conditions, and updated X milestone dictionaries
            fitness_current, done, x_pos_flags_points, x_pos_flags_ratios = adjust_fitness(fitness_current=fitness_current, 
                            done=done,
                            frame=frame, 
                            x_pos_last=x_pos_last, 
                            x_pos_current=x_pos_current, 
                            current_max_x_pos=current_max_x_pos, 
                            screen_x_end=screen_x_end, 
                            x_pos_flags_ratios=x_pos_flags_ratios,
                            x_pos_flags_points=x_pos_flags_points, 
                            points_flag=False, # Whether or not to award points based on points dict
                            ratio_flag=True # Whether or not to award points based on ratio dict
                            )
            ### Update the previous X position and the max X position
            x_pos_last = x_pos_current
            current_max_x_pos = max(x_pos_current, current_max_x_pos)
            
            ### If fitness does increase, reset counter and update max fitness variable
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                fitness_incr_counter = 0
            else:
                fitness_incr_counter += 1
                

            """
            Number of steps Sonic can take without progressing in the X direction before done == 'True'
            """
            in_place_limit = 5000
            """
            Number of steps Sonic can take without increasing fitness before done == 'True'
            """
            fitness_incr_limit = 5000
            """
            X value at which the counter limits will be scaled to be larger. 
            Allows Sonic to have more chances to try things later in the run. 
            """
            modify_limit_val = 100
            """
            Factor to scale by
            Minimum value will be 0.3 and after that, it'll be a ratio of
                (current max / end x value)
            """
            round_expire_scale = max(0.3, current_max_x_pos / screen_x_end)
            
            ### Update done condition using the done_conditions function
            done = done_conditions(lives_left, done,
                    x_pos_current, modify_limit_val,
                    fitness_incr_counter, fitness_incr_limit, 
                    steps_in_place, in_place_limit, round_expire_scale)
            ### If done, print info about the genome and the fitness it achieved
            if done:
                print("\tGenome id : ", genome_id, "// fitness : ", fitness_current, "\n")
            
            ### Set genome's fitness to the current fitness
            genome.fitness = fitness_current

        
 


def run(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    ### Load checkpoint population if resume == 'True'
    if resume == True:
        p = neat.Checkpointer.restore_checkpoint(start_checkpoint_file)
    else:
        p = neat.Population(config)
    
    ### Add some reporters to the generation to get data about them midrun
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10, filename_prefix = checkpoint_prefix))
    winner = p.run(eval_genomes)

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)


if __name__ == '__main__':
    ### Original author's note below:
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_SK3_neat')
    run(config_path)
