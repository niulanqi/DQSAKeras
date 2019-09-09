
class config:
    N = 3  # number of users
    K = 1  # number of channels
    TimeSlots = 50
    Actions = K + 1
    batch_size = 16
    M = batch_size
    train_iterations = 1
    usrTimeSteps = 1
    input_size_user = (1, usrTimeSteps, 2 * K + 2)  # the paper state that there is only (2K+2) X 100
    central_first_axis = batch_size * N
    input_size_central = (central_first_axis, None, 2 * K + 2)
    model_path = r'saved_models/network_central_add_grad'
    log_dir = r'log_dir_add_grad/'
    ckpt_path =r'/home/dorliv/Desktop/DQSAKeras/saved_models/network_central_add_grad/checkpoint'
    Iterations = 10000
    Episodes = 100
    memory_size = 10000
    Gamma = 0.95
    useUserHistory = False
    LstmUnits = 100
    debug_freq = 10

    @staticmethod
    def learning_rate_schedule(epoch: int):
        if epoch <= 50:
            return 5e-3
        if 50 < epoch <= 250:
            return 1e-4
        if 250 < epoch <= 500:
            return 5e-5
        if 500 < epoch <= 750:
            return 1e-5
        if 750 < epoch <= 1000:
            return 5e-6
        if 1000 < epoch <= 1500:
            return 1e-6
        if 1500 < epoch <= 2500:
            return 5e-7
        else:
            return 1e-7


    @staticmethod
    def temperature_schedule(beta):
        # temperature starts at 1 and grows to 20 as the iterations increase
        # every 100 iterations we increase beta by 1
        return min(beta + 0.05, 20)
