# NEAT-python-Sonic

## Goal

Implemente the NEAT reinforcement learning algorithm in Sonic The Hedgehog 3.

## High level flow

Running the `main_SK3_neat.py` file from the command line will start the training. 

As the game runs, each frame is processed by NEAT's neural network which predicts what the next button push should be. Each frame is also analyzed to adjust the `fitness` value for the run and the `done` conditions.

### Fitness Value

Each run gets a fitness value that is adjusted as the agent plays through the level. Adjustments to the fitness value are made by the `fitness_change_SK3_neat.py` file. 

All factors currently implemented that will impact the fitness are listed below.
- Every 100 and 500 frames the agent loses fitness.
- If the absolute change in x value is < 2, the agent loses fitness.
- If the agent achieves a new max X value, they gain fitness.
- If the agent passes certain milestones they are awarded fitness.

### Done conditions

Each run has a done boolean that is used in a `While` loop which runs a bulk of the program. This loop and the run are terminated when this boolean is set to `True`.

In the real world, a done condition would probably just be the agent's death but due to time constrants, its useful to cut off unnsuccessful agents at some point.

All the possible situations which will convert the `done` value to true are listed below :
- If agent loses a life; done.
- If the agent hasn't progressed in the X direction in a certain number of steps.
- If the agent's fitness hasn't increased in a certain number of steps. 
- If the agent reaches the end of the level. 

## To do

- Adjust fitness conditions
    - Agents have negative fitness right off the bat so maybe add a certain baseline beginning fitness.
        - Could make it a function of the amount of time before `done` == True so agents who don't move would have 0 when they expire.
    - Maybe reward agent for being close to max X value rather than simply exceeding it.
        - This would allow for reward even when not RIGHT at the max X value which would emphasize exploration over just pushing right.
    - Work on the milestone fitness rewards
        - Feels unnatural for the coder to have input on the relative value of certain arbitrary milestones.
            - Seems like external input that the agent shouldn't have access to.
            - Could be good to have distances be predetermined fractions of the distance the agent has to run rather than hardcoded values.
                - Change (+reward at X == 1000), to (+reward at (1/5)*total_distance)
        - A scaling factor can have weird impacts on negative fitness agents.
- Adjust done conditions
    - Modify limit seems weirdly artificial.
        - Try to scale that value between beginning and end of level more smoothly.
    - Limiting steps in place and forcing forward progress could be preventing exploration. 
        - Would need to increase or remove limits to test.
- Try on different games
- Try on different levels of Sonic 3
- Mess around with config settings more. 
    - Made small adjustments but its difficult to quantify the effects of a change to the config.
