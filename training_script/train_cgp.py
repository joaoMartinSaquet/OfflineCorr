import sys
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import pprint




file_path = os.path.abspath(__file__)
script_directory = os.path.dirname(file_path)
print("script directory : ", script_directory)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from utils import dataset_handling
from utils.dataset_handling import *
from utils.model_handling import *
from pyCGP.pycgp import cgp, evaluators, viz, cgpfunctions
from pyCGP.pycgp.cgp import *
from pyCGP.pycgp.cgpfunctions import *

OPTIMIZE_HYPERPARAMETERS = True

corrector_lib0 =  [CGPFunc(f_sum, 'sum', 2, 0, '+'),
            CGPFunc(f_aminus, 'aminus', 2, 0, '-'),
            CGPFunc(f_mult, 'mult', 2, 0, '*'),
            CGPFunc(f_sin, 'sin', 1, 0, 'sin'),
            CGPFunc(f_cos, 'cos', 1, 0, 'cos'),
            CGPFunc(f_div, 'div', 2, 0, '/'),
            CGPFunc(f_const, 'c', 0, 1, 'c')
            ]   
            # CGPFunc(f_const, 'c', 0, 1, 'c')

corrector_lib1 =  [CGPFunc(f_sum, 'sum', 2, 0, '+'),
            CGPFunc(f_aminus, 'aminus', 2, 0, '-'),
            CGPFunc(f_mult, 'mult', 2, 0, '*'),
            CGPFunc(f_div, 'div', 2, 0, '/'),
            CGPFunc(f_gt, 'gt', 1, 0, '>'),
            CGPFunc(f_const, 'c', 0, 1, 'c')
            ]

corrector_lib2 =  [CGPFunc(f_sum, 'sum', 2, 0, '+'),
            CGPFunc(f_aminus, 'aminus', 2, 0, '-'),
            CGPFunc(f_mult, 'mult', 2, 0, '*'),
            CGPFunc(f_div, 'div', 2, 0, '/'),
            CGPFunc(f_gt, 'gt', 2, 0, '>'),
            CGPFunc(f_lt, 'lt', 2, 0, '<'),
            CGPFunc(f_log, 'log', 1, 0, 'log'),
            CGPFunc(f_const, 'c', 0, 1, 'c')
            ]


LIBRARY = [corrector_lib0, corrector_lib1, corrector_lib2]

# wandb stuff
sweep_config = {
    'method': 'random'
    }

metric = {
    'name': 'loss',
    'goal': 'maximize'   
    }

sweep_config['metric'] = metric


parameters_dict = {
    'col': {
        'values': [30, 60, 120]
        },
    'mu': {
          'values': [30, 40, 50, 60]
        },
    'lbd': {
          'values': [60, 80, 100]
        },
    
    'library' : {
        'values': [1, 2]
            }
    }

sweep_config['parameters'] = parameters_dict

def train_cgp(config = None):
    # TODO implement that https://docs.wandb.ai/tutorials/sweeps/
    
    config = load_config("config/cgp_config.yaml")
    hyperparameters = config['hyperparameters']
    log_dir = f"/home/jmartinsaquet/Documents/code/IA2_codes/OfflineCorr/results/wandb/CGP"

    # prepare data
    # x, y, _ = read_dataset(f"/home/jmartinsaquet/Documents/code/IA2_codes/clone/datasets/{experiment_name}.csv", "vec", 0, True)
    xd, yd, _ = read_dataset(f"/home/jmartinsaquet/Documents/code/IA2_codes/clone/datasets/{experiment_name}.csv", "vec")
    input_name = list(xd.columns)
    # x = x.to_numpy()
    # y = y.to_numpy()
    x, yt, scaler_x = preprocess_dataset(xd, yd, 'minmax')


    with wandb.init(config=config):
        config = wandb.config
        print("config : ", config)
        n_input = x[0].shape[0]
        fitts_evaluator = evaluators.SREvaluator(x_train=x, y_train=y, n_inputs=n_input, n_outputs=2,
                                             col=config.col,
                                             row=1, library=LIBRARY[config.library], loss='mse')
    
        # print("x.")
        hof, hist = fitts_evaluator.evolve(mu =config.mu, nb_ind=config.lbd, num_csts=10,
                                        mutation_rate_nodes=hyperparameters['m_node'], mutation_rate_outputs=hyperparameters['m_output'],
                                        mutation_rate_const_params=hyperparameters['m_const'], n_it=hyperparameters['n_gen'], folder_name=log_dir)

        wandb.log({"loss": hist[-1]})

if __name__ == '__main__':

    experiment_name = "P0_C0"
    print("---------------- {} ----------------".format(experiment_name))
    if OPTIMIZE_HYPERPARAMETERS:
        wandb.login()
        pprint.pprint(sweep_config)
        sweep_id = wandb.sweep(sweep_config, project="cgp_opt")
        wandb.agent(sweep_id, train_cgp, count=10)
    else:
        log_dir = f"../results/{experiment_name}/CGP/"
        print("logging to : ", log_dir)
        os.makedirs(log_dir, exist_ok=True)


        # load dataset
        # xd, yd, _ = read_dataset(f"/home/jmartinsaquet/Documents/code/IA2_codes/clone/datasets/{experiment_name}.csv", "vec")
        # x, y, _ = read_dataset(f"/home/jmartinsaquet/Documents/code/IA2_codes/clone/datasets/{experiment_name}.csv", "vec", 0, True)
        xd, yd, _ = read_dataset(f"/home/jmartinsaquet/Documents/code/IA2_codes/clone/datasets/{experiment_name}.csv", "vec")
        input_name = list(xd.columns)
        # x = x.to_numpy()
        # y = y.to_numpy()
        x, yt, scaler_x = preprocess_dataset(xd, yd, 'minmax')
        
        # convert to array
        # x = np.array(x)
        # y = np.array(y)
        y, xt, scaler_y = preprocess_dataset(yd, xd, 'minmax')

        # read hyperparameters
        config = load_config("config/cgp_config.yaml")
        hyperparameters = config['hyperparameters']
        print("hyperparameters: ")
        for key, value in hyperparameters.items():
            print(f"{key}: {value}")

        # create library
        library = corrector_lib2 #[cgp.CGPFunc()] # None for now (we can see later for customs)
        n_input = x[0].shape[0]
        fitts_evaluator = evaluators.SREvaluator(x_train=x, y_train=y, n_inputs=n_input, n_outputs=2,
                                                col=hyperparameters['col'],
                                                row=hyperparameters['row'], library=library)
        
        # print("x.")
        hof, hist = fitts_evaluator.evolve(mu = hyperparameters['mu'], nb_ind= hyperparameters['lambda'], num_csts=10,
                                            mutation_rate_nodes=hyperparameters['m_node'], mutation_rate_outputs=hyperparameters['m_output'],
                                            mutation_rate_const_params=hyperparameters['m_const'], n_it=hyperparameters['n_gen'], folder_name=log_dir)
        
        # input_name = ["x", "y", "dx", "dy"]
        output_name = ["dxe", "dye"]

        # save the best one 
        hof.save(log_dir+"hof.log")

        
        y_pred = hof.run(x).T
        eq = ''
        try : 
            eq = fitts_evaluator.best_logs(input_name, output_name)
        except: 
            pass


        # y = y.to_numpy()        
        print("y pred ! ", y_pred)
        print("equation : ", eq)
        plt.figure()

        plt.plot(y[:, 0], y[:, 1], '.', label = "true")
        # plt.plot(x[:,0], x[:,1], '.', label = "input")
        plt.plot(y_pred[:,0], y_pred[:,1], '.', label = "pred")
        plt.title("pred vs true")
        plt.legend()
        
        # plt.figure()
        # plt.plot(x[:,2], x[:,3], '.', label = "input")
        # plt.plot(y_pred[:,0], y_pred[:,1], '.', label = "pred")
        # plt.legend()
        
        # plt.figure()
        # plt.plot(x[:,2], x[:,3], '.', label = "input")
        # plt.plot(y[:,0], y[:,1], '.', label = "pred")
        # plt.title("input vs true")
        # plt.legend()
        
        hof_graph = hof.netx_graph(input_name, output_name, True, False, False)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        viz.draw_net(ax, hof_graph, n_input, 2)
        plt.show()