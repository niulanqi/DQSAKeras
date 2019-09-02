
class config:
    N = 3  # number of users
    K = 1  # number of channels
    TimeSlots = 75
    Actions = K + 1
    batch_size = 16
    M = 18
    train_iterations = 1
    usrTimeSteps = 1
    input_size_user = (1, usrTimeSteps, 2 * K + 2)  # the paper state that there is only (2K+2) X 100
    central_first_axis = batch_size * N
    input_size_central = (central_first_axis, None, 2 * K + 2)
    # training_freq = 15
    # multiplication so we believe there is only one time step0
    output_size_central = (central_first_axis, Actions)
    output_size = (None, Actions)
    model_path = r'saved_models/network_central'
    log_dir = r'log_dir/'
    ckpt_path =r'/home/dorliv/Desktop/DQSAKeras/saved_models/network_central/checkpoint.hdf5'
    Iterations = 10000
    Episodes = 80
    memory_size = 10000
    Gamma = 0.95
    useUserHistory = False
    LstmUnits = 100
    debug_freq = 10

    @staticmethod
    def learning_rate_schedule(epoch: int):
        if epoch <= 10:
            return 1e-2
        if 10 < epoch <= 250:
            return 5e-3
        if 250 < epoch <= 500:
            return 1e-3
        if 500 < epoch <= 750:
            return 5e-4
        if 750 < epoch <= 1500:
            return 1e-4
        if 1500 < epoch <= 2500:
            return 5e-5
        if 2500 < epoch <= 3500:
            return 1e-5
        else:
            return 5e-6


    @staticmethod
    def temperature_schedule(beta):
        # temperature starts at 1 and grows to 20 as the iterations increase
        # every 200 iterations we increase beta by 1
        return min(beta + 0.005, 20)
