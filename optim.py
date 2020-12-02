from client import main
import argparse
from hyperopt import tpe, hp, fmin
import config

parser = argparse.ArgumentParser(description='Searching server')
parser.add_argument('-p', '--port', action='store', type=int, default=config.port,
                    help='TCP port')
parser.add_argument('--stop', action='store', type=int, default=config.stop,
                    help='Stop early')
parser.add_argument('-s', '--server', action='store', type=str, default=config.server,
                    help='TCP port')
parser.add_argument('-d', '--dataset', action='store', type=str, default=config.dataset,
                    help='TCP port')
parser.add_argument('-l', '--langSearch', action='store', type=str, default=config.langSearch,
                    help='Context language')
parser.add_argument('-q', '--langQuestion', action='store', type=str, default=config.langQuestion,
                    help='Question language')
parser.add_argument('-f', '--save-as', action='store', type=str, default="",
                    help='Save result')
parser.add_argument('-k', '--topk', action='store', type=int, default=config.topk,
                    help='Retriev k contexts')
parser.add_argument('-n', '--dry-run', action='store_true',
                    help='Print configuration')
parser.add_argument('-b', action='store', type=int, default=config.b,
                    help='Stop early')
parser.add_argument('--k1', action='store', type=int, default=config.k1,
                    help='Stop early')
args = parser.parse_args()

def objective(params):
    args.b=params['b']
    args.k1=params['k1']
    args.save_as = "b{}_k{}_en_en.json".format(params['b'],params['k1'])
    print("b: {} k1: {}".format(params['b'],params['k1']))
    return -main(args)

space = {
        'b': hp.uniform('b', 0., 1.),
        'k1': hp.uniform('k1', 0., 1.5)
        }

best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50
        )

print(best)
