import argparse


def get_main_options():

    # model parameters
    model_parser = argparse.ArgumentParser(description='Model Parameter')
    model_parser.add_argument('--dataset',metavar='', type=str, default='toy_lgssm')
    model_parser.add_argument('--model', metavar = '', type=str, default='VRNN-Gauss-I')
    model_parser.add_argument('--do_train',metavar = '',  type=bool, default=True)
    model_parser.add_argument('--do_test', metavar = '', type=bool, default=True)
    model_parser.add_argument('--logdir',metavar = '',  type=str, default='same_dataset')
    model_parser.add_argument('--normalize', metavar = '', type=bool, default=True)
    model_parser.add_argument('--seed', metavar = '', type=int, default=1234)
    model_parser.add_argument('--optim', metavar = '', type=str, default='Adam')
    model_parser.add_argument('--showfig', metavar = '', type=bool, default=False)
    model_parser.add_argument('--savefig', metavar = '', type=bool, default=False)
    model_parser.add_argument('--known_parameter', metavar = '', type=str, default='None')
 
    model_options = model_parser.parse_args()
    print("hihi",model_options.model)
    return model_options