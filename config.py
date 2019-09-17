
class config:
    N = 3  # number of users
    K = 1  # number of channels
    TimeSlots = 50
    TimeSlotsEvaluate = 200
    Actions = K + 1
    batch_size = 8
    M = batch_size * 3
    train_iterations = 1
    usrTimeSteps = 1
    input_size_user = (1, usrTimeSteps, 2 * K + 2)  # the paper state that there is only (2K+2) X 100
    central_first_axis = batch_size * N
    input_size_central = (None, None, 2 * K + 2)
    model_path = r'saved_models/network_central_continue'
    log_dir = r'log_dir/'
    evaluate_log_dir=r'log_dir'
    ckpt_path =r'/home/dorliv/Desktop/DQSAKeras/saved_models/network_central/checkpoint'
    load_ckpt_path = r'/home/dorliv/Desktop/DQSAKeras/saved_models/network_central/checkpoint'
    Iterations = 10000
    Episodes = 100
    memory_size = 10000
    Gamma = 0.95
    LstmUnits = 100
    debug_freq = 10
    learning_rate_multiplier = 1e-0

    @staticmethod
    def learning_rate_schedule(epoch: int):
        if epoch <= 250:
            return 5e-3
        if 250 < epoch <= 500:
            return 1e-3
        if 500 < epoch <= 1500:
            return 5e-4
        if 1500 < epoch <= 2000:
            return 1e-4
        if 2000 < epoch <= 2500:
            return 5e-5
        if 2500 < epoch <= 3000:
            return 1e-5
        if 3000 < epoch <= 3500:
            return 5e-6
        if 3500 < epoch <= 4000:
            return 1e-6
        if 4000 < epoch <= 4500:
            return 5e-7
        else:
            return 1e-7


    @staticmethod
    def temperature_schedule(beta):
        # temperature starts at 1 and grows to 20 as the iterations increase
        # every 100 iterations we increase beta by 1
        return min(beta + 0.01, 25)
