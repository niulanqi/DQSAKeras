# DQSAKeras
DQSA in tensorflow 2.0

# Walkthrough:
1) *Environment.py* - Environment.py includes a wrapper class for Dynamic Spectrum Access scenario,
                   where each user decides if to remain idle or to transmit ( action 0 means idle otherwise it means the user chose to transmit)
                   the "Env" class is a basic class that holds the inner state of the channels and updates the inner state based on what the users chose, if only one user chose to transmit on channel k then it will get an ACK signal else it will get 0 ACK
                   the inner state is restarted through the "reset" method and it is automatically restarted each time step by the "getNextState" method.
                   the "OneTimeStepEnv" inherits for the "Env" class and just organizes the data for the user's interface with the DQSA
                   
2) *Memory.py* - this module consist of the two main classes
                  2.1) "Memory": a data structure that by the time the episode ends, holds (state, next_state, reward, action) for each time step
                                 it also has the ability to output a part of the sequence through "withdrawSequenceOfExperience" method 
                  2.2) ""Experience Replay": a data structure that holds "Memory" classes, it is used in the training phase when we want to get a batch of Memories
                                             it has the ability to flush the memories it currently holds through the "flush" method
3) *model.py* - this module holds the DQSA model, the important method in this module is the "fit" method, it decides the optimizer learning rate
                and extracts batch size of memories from the ER, afterwards, we evaluate the target vector for each time step and calculate the gradients(without updating the policy)
                after we calculated the gradients for each time step we apply the gradients(update the policy)
4) *train.py* - the main program, we first create the user and central network and then we start the main function called "trainDqsa".
                in the "trainDqsa" function we follow the algorithm in the DQSA paper with the exception of choosing the action ( we use the formula that is given to us in the matlab code).
                all the users have identical stateful DQSA, at each time step each user evaluates an action through the DQSA (using the state from the last timestep
                and the stateful capability of the user's DQSA net) and executes the action through the environment, after all the users executes an action we extract the next state from the 
                environment, save it in a "Memory" type data structre and proceed to the next time-step.
                at the end of an episode, we insert the full "Memory" type data structre to the Experience replay for later use in the training pahse.
                every M episodes we execute a training phase routine, which uses the model fit method, afterwards we flush the Experience replay
                to clear the old policy experiences and update the policy of the user networks through "synchWithCentral" function.
 
