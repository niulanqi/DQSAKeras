import tensorflow as tf
from tensorflow.python.keras import callbacks
from config import config
import os
from model import DQSAVersion2
from logger_utils import get_logger
from Environment import OneTimeStepEnv
import numpy as np
from train import initCTP, getAction
import plotly.graph_objects as go



def draw_heatmap(heatmap):
    fig = go.Figure(data=go.Heatmap(
        z=np.asarray(heatmap).T))
    fig.update_layout(
        title=go.layout.Title(
            text="User action",
            xref="paper",
            x=0
        ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text="Time Slots",
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="#7f7f7f"
                )
            )
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text="Users",
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="#7f7f7f"
                )
            )
        )
    )
    fig.show()


if __name__ == '__main__':
    userNet = DQSAVersion2(input_size=config.input_size_user, usernet=True, batch_size=config.N)
    userNet.load_weights(path="/home/dorliv/Desktop/DQSAKeras/saved_models/network_central_best_dynamic_users/checkpoint")
    logger = get_logger(os.path.join(config.evaluate_log_dir, "evaluate_log"))
    Tensorcallback = callbacks.TensorBoard(config.evaluate_log_dir,
                                           write_graph=True, write_images=False)
    Tensorcallback.set_model(userNet.model)
    env = OneTimeStepEnv()
    beta = 10
    alpha = 0  # e_greedy
    draw_heatmap_flag = True
    channelThroughPutPerTstep = initCTP(config.TimeSlotsEvaluate)  # init the data structure to view the mean reward at each t
    for iteration in range(config.evaluate_iterations):
        channelThroughPutMean = 0
        loss_value = []
        collisonsMean = 0
        idle_timesMean = 0
        for episode in range(config.Episodes):
            heatmap = []
            collisons = 0
            idle_times = 0
            channelThroughPut = 0
            Xt = env.reset()
            # Xt = np.expand_dims(Xt, axis=1)
            for tstep in range(config.TimeSlotsEvaluate):
                # ----- start time-steps loop -----
                UserQvalues = userNet(Xt)
                actionThatUsersChose = [getAction(Qvalues=UserQvalue, temperature=beta, alpha=alpha) for UserQvalue in UserQvalues]
                for usr in range(config.N):
                    env.step(action=actionThatUsersChose[usr], user=usr)
                heatmap.append(actionThatUsersChose)
                nextStateForEachUser, rewardForEachUser, ack_vector = env.getNextState()  # also resets the env for the next t step
                Xt = nextStateForEachUser  # state = next_State
                collisons += env.collisions
                idle_times += env.idle_times
                ack_sum = np.sum(
                    ack_vector)  # rewards are calculated by the ACK signal as competitive reward mechanism
                tmp = channelThroughPutPerTstep[tstep + 1]
                channelThroughPutPerTstep[tstep + 1].append(ack_sum)
                channelThroughPut += ack_sum
            if draw_heatmap_flag:
                draw_heatmap(heatmap=heatmap)
            userNet.reset_states()
            collisonsMean += collisons / config.TimeSlotsEvaluate
            idle_timesMean += idle_times / config.TimeSlotsEvaluate
            channelThroughPutMean += channelThroughPut / config.TimeSlotsEvaluate
            print(
                "Iteration {}/{}- Episode {}/{}:  collisions {}, idle_times {}, channelThroughput is {}"
                    .format(iteration, config.evaluate_iterations, episode, config.Episodes, collisons / config.TimeSlotsEvaluate,
                            idle_times / config.TimeSlotsEvaluate, channelThroughPut / config.TimeSlotsEvaluate))
            logger.info(
                "Iteration {}/{}- Episode {}/{}:  collisions {}, idle_times {}, channelThroughput is {}"
                    .format(iteration, config.evaluate_iterations, episode, config.Episodes, collisons / config.TimeSlotsEvaluate,
                            idle_times / config.TimeSlotsEvaluate, channelThroughPut / config.TimeSlotsEvaluate))

        channelThroughPutPerTstep = [np.mean(x) for x in channelThroughPutPerTstep]
        for i, x in enumerate(channelThroughPutPerTstep):
            logs = {'channelThroughputTstep': x}
            Tensorcallback.on_epoch_end(epoch=iteration * (config.TimeSlotsEvaluate + 2) + i, logs=logs)
        channelThroughPutPerTstep = initCTP(config.TimeSlotsEvaluate)  # init the data structure
        print(
            "Iteration {}/{}  collisions {}, idle_times {}, channelThroughput is {}"
                .format(iteration, config.evaluate_iterations, collisonsMean / config.Episodes,
                        idle_timesMean / config.Episodes, channelThroughPutMean / config.Episodes))
        logger.info(
            "Iteration {}/{}  collisions {}, idle_times {}, channelThroughput is {}"
                .format(iteration, config.evaluate_iterations, collisonsMean / config.Episodes,
                        idle_timesMean / config.Episodes, channelThroughPutMean / config.Episodes))


