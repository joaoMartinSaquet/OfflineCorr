import sys
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt


# from utils import dataset_handling
from utils.dataset_handling import *
from utils.model_handling import *
from pyCGP.pycgp import cgp, evaluators, viz, cgpfunctions


def f_id(args, const_params):
    return x

if __name__ == '__main__':

    experiment_name = "P0_C0"
    print("---------------- {} ----------------".format(experiment_name))
    log_dir = f"../results/{experiment_name}/CGP/"
    print("logging to : ", log_dir)
    os.makedirs(log_dir, exist_ok=True)


    # load dataset
    x, y, _ = read_dataset(f"/home/jmartinsaquet/Documents/code/IA2_codes/clone/datasets/{experiment_name}.csv", "vec")
    
    # convert to array
    x = np.array(x)
    y = np.array(y)


    # read hyperparameters
    config = load_config("config/cgp_config.yaml")
    hyperparameters = config['hyperparameters']
    print("hyperparameters: ")
    for key, value in hyperparameters.items():
        print(f"{key}: {value}")

    # create library
    library = [
            cgp.CGPFunc(cgpfunctions.f_sum, 'sum', 2, 0, '+'),
            cgp.CGPFunc(cgpfunctions.f_aminus, 'aminus', 2, 0, '-'),
            cgp.CGPFunc(cgpfunctions.f_mult, 'mult', 2, 0, '*'),
            # cgp.CGPFunc(f_id, "identity", 1, 0, 'I'),
            # cgp.CGPFunc(cgpfunctions.f_sin, 'sin', 1, 0, 'sin'),
            # cgp.CGPFunc(cgpfunctions.f_cos, 'cos', 1, 0, 'cos'),
            cgp.CGPFunc(cgpfunctions.f_div, 'div', 2, 0, '/'),
            cgp.CGPFunc(cgpfunctions.f_const, 'c', 0, 1, 'c')
            ] #[cgp.CGPFunc()] # None for now (we can see later for customs)
    n_input = x[0].shape[0]
    fitts_evaluator = evaluators.SREvaluator(x_train=x, y_train=y, n_inputs=n_input, n_outputs=2,
                                             col=hyperparameters['col'],
                                             row=hyperparameters['row'], library=library)
    
    # print("x.")
    hof, hist = fitts_evaluator.evolve(mu = hyperparameters['mu'], nb_ind= hyperparameters['lambda'], num_csts=3,
                                        mutation_rate_nodes=hyperparameters['m_node'], mutation_rate_outputs=hyperparameters['m_output'],
                                        mutation_rate_const_params=hyperparameters['m_const'], n_it=hyperparameters['n_gen'], folder_name=log_dir)
    
    input_name = ["x", "y", "dx", "dy"]
    output_name = ["dxe", "dye"]

    # save the best one 
    hof.save(log_dir+"hof.log")

    
    y_pred = hof.run(x).T
    eq = ''
    try : 
        eq = fitts_evaluator.best_logs(input_name, output_name)
    except: 
        pass
    
    
    print("y pred ! ", y_pred.shape)
    print("equation : ", eq)
    plt.figure()
    plt.plot(y_pred[:,0], y_pred[:,1], '.', label = "pred")
    plt.plot(y[:, 0], y[:, 1], '.', label = "true")
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