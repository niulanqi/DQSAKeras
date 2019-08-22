
class config:
    N = 3  # number of users
    K = 1  # number of channels
    TimeSlots = 100
    Actions = K + 1
    batch_size = 5
    M = 20
    train_iterations = 4
    usrTimeSteps = 1
    input_size_user = (1, usrTimeSteps, 2 * K + 2)  # the paper state that there is only (2K+2) X 100
    central_first_axis = batch_size * N
    input_size_central = (central_first_axis, None, 2 * K + 2)
    # training_freq = 15
    # multiplication so we believe there is only one time step0
    output_size_central = (central_first_axis, Actions)
    output_size = (None, Actions)
    model_path = r'saved_models/network_central_history'
    log_dir = r'log_dir_history/'
    ckpt_path =r'/home/dorliv/Desktop/DQSAKeras/saved_models/network_central_history/checkpoint.hdf5'
    Iterations = 10000
    Episodes = 80
    memory_size = 10000
    Gamma = 0.95
    LstmUnits = 100

    @staticmethod
    def learning_rate_schedule(epoch: int):
        if epoch <= 300:
            return 1e-4
        if 300 < epoch <= 500:
            return 5e-5
        if 500 < epoch <= 1000:
            return 1e-5
        if 1000 < epoch <= 2500:
            return 5e-6
        if 2500 < epoch <= 5000:
            return 1e-6
        if 5000 < epoch <= 7500:
            return 1e-6
        else:
            return 5e-7


    @staticmethod
    def temperature_schedule(iteration: int):
        # temperature starts at 1 and grows to 20 as the iterations increase
        return (1 - iteration / config.Iterations) * 1 + 20 * (iteration / config.Iterations)
