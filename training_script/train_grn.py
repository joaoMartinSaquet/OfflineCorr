import asyncio
import websockets
import json
import numpy as np
from sklearn.metrics import mean_squared_error as mse
import os
import sys

file_path = os.path.abspath(__file__)
script_directory = os.path.dirname(file_path)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_handling import read_dataset, preprocess_dataset

class Regressor(object):

    def __init__(self, fitness_function = 'mse'):

        if fitness_function == 'mse':
            self.fit_fun = mse

        # defaults datas 
        self.x = np.array([np.linspace(-5,5), np.linspace(-5,5)]).T
        self.y = self.x

    def load_data(self, x, y):
        """load data from a script

        Args:
            x (array): GRN inputs
            y (array): true output to get from GRN 
        """

        self.x = x
        self.y = y

    async def handler(self, websocket):
        """This function handle the communication between GRNEAT and the python evaluator 
        
            Python Component (py)                       Java Component
            -----------------------------------------------------------------
            1. Sends `GRN_input`                  ----> 1. Receives `GRN_input`
                                                            - Processes `GRN_input` to generate `yGRN`

            2. Waits to receive `yGRN`             <--- 2. Sends `yGRN`
            - Computes the fitness based on `yGRN`

            3. Sends the computed fitness            -> 3. Receives the computed fitness
        Args:
            websocket (_type_): _description_
            path (_type_): _description_
        """
        self.end = False
        while True:
            # attend un message
            msg = await websocket.recv()

            if msg == "START":
                reply = {"inputs" : self.x.tolist()}
            elif msg == "END":
                reply = "ended"
                self.end = True
            else : 
                # get ygrn
                y_grn = json.loads(msg)['y_grn']
                print(f"y_grn  min {min(y_grn)}, max {max(y_grn)}" )
                reply = {"fitness" : -self.fit_fun(y_grn, self.y)}    
                print("reply : ", reply)
            await websocket.send(json.dumps(reply))
            if self.end:
                break
        

    async def start_server(self):
        
        # start_server = websockets.serve(self.handler, "localhost", config[server][port])

        async with websockets.serve(self.handler, "localhost", 8000):
        # async with websockets.serve(self.handler, host, port):
            print(f"WebSocket server started at ")
            await asyncio.Future()  # Run forever



if __name__ == "__main__":
    experiment_name = "P0_C0"
    print("---------------- {} ----------------".format(experiment_name))
    print("GRN trainign")
    log_dir = f"../results/{experiment_name}/GRN/"
    print("logging to : ", log_dir)
    os.makedirs(log_dir, exist_ok=True)


    # load dataset
    x, y, _ = read_dataset(f"/home/jmartinsaquet/Documents/code/IA2_codes/clone/datasets/{experiment_name}.csv", "vec")
    
    # process the dataset to be in range [-1;1]   
    xt, y, scaler = preprocess_dataset(x, y, 'minmax')
    yt, x, scaler = preprocess_dataset(y, x, 'minmax')

    server = Regressor()
    server.load_data(xt, yt)
    asyncio.run(server.start_server())
    server.start_server()