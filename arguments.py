import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='real-time inference')
    # General arguments
    parser.add_argument('-v', '--view', type=str, default='side', help='which view of ITOP dataset')
    parser.add_argument('-m', '--mode', type=str, default='test', help='train or test')
    parser.add_argument('-i', '--start_idx', type=int, default=32, help='start from xth frame')
    parser.add_argument('--network', type=str, default='cnn', help='choose failure recovery network')
    # Model arguments
    parser.add_argument('--rtw', type=str, default='1026', help='which rtw model to use')
    parser.add_argument('--cnn', type=str, default='20201026', help='which cnn model to use')
    parser.add_argument('-arc', '--architecture', type=str, default='3,3', help='filter widths separated by comma')
    parser.add_argument('-e', '--epoch', type=int, default=80, help='which epoch result of cnn to load')
    parser.add_argument('--nn', type=str, default='model_parameters_nn', help='which nn model to use')
    parser.add_argument('--n_input', type=int, default=30*3, help='input size of NN')
    parser.add_argument('--n_hidden', type=int, default=1024, help='hidden channels of NN')
    # Experimental arguments
    parser.add_argument('-n', '--n-steps', type=int, default=96, help='num of steps in rtw')
    parser.add_argument('-s', '--step-size', type=int, default=2, help='step size in rtw')
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--channels', type=int, default=1024)
    args = parser.parse_args()

    return args