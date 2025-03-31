from pynput import mouse, keyboard
import numpy as np
import matplotlib.pyplot as plt
import torch
import time

'''
This script is a script that is going to load the trained model 
    - ANN
    - LSTM
    - CGP
    - GRN

and be tested on the user directly to see if there is any differences between with and without the controller
'''

class Corrector(object):
    """Controller with the model"""

    def __init__(self, model, device):

        self.corrector = model.eval()
        self.x_poses = []
        self.y_poses = []


        self.device = device
        # get the mouse
        self.mouse = mouse.Controller()
        self.x_poses.append(self.mouse.position[0])
        self.y_poses.append(self.mouse.position[1])

        self.min_input = np.array([0, 0, -80, -80, 0])
        self.max_input = np.array([1920, 1080, 80, 80, 500])

        self.flag = True # we can modify the mouse position 

        self.dxs = [0.]
        self.dys = [0.]

        self.t = time.time()
        self.k = 0

        self.correct = False
        print("Controller ready to be used !")
        print("Correction activated ? ", self.correct)


    def main(self):

        self.i = 0
        # do it still forever
        while True:
            print("self.i ", self.i)
            if self.i == 1000:
                print("move ! ")
                mouse.position = (0, 0)
            self.i += 1
            # self.mouse.move(10,20)

                # self.flag = False

    
    def on_move(self, x, y):


        # self.flag = True
        # print("x : ", x, " y : ", y, " i : ", self.k)
        # self.k += 1
        # # get the prediction
        if self.flag:
            self.flag = not self.flag
            
            t_start = time.time()

            
            
            dx = x - self.x_poses[-1]
            dy = y - self.y_poses[-1]
            # # mag = np.sqrt(dx**2 + dy**2)

            dt = time.time() - self.t

            self.dxs.append(dx)
            self.dys.append(dy)

            model_input = np.array([[self.x_poses[-1], self.y_poses[-1], dx, dy, dt]])
            model_input = (model_input - self.min_input) / (self.max_input - self.min_input) * 2 - 1 

            model_input = torch.from_numpy(model_input).float()
            # model_input = torch.tensor(model_input, dtype=torch.float)

            # # get the prediction
            with torch.inference_mode():
                prediction = self.corrector.forward(model_input)


            new_cursor_position = np.round((self.x_poses[-1], self.y_poses[-1]) + prediction.cpu().detach().numpy())
            self.mouse.position = (new_cursor_position[0][0], new_cursor_position[0][1])

            self.x_poses.append(x)
            self.y_poses.append(y)
            
            # # print("flag : , i :", self.flag, self.k) 
            # if self.correct and self.flag and dx == 0 and dy == 0: 
            #     self.mouse.move(np.round(new_cursor_position[0]), np.round(new_cursor_position[1])) # this is triggering a new event...
            #     print(" correction off !" )
            #     self.flag = not self.flag
            #     self.k = 0
            #     # time.sleep(0.2)
            # elif not self.flag and self.k >10:
            #     print(" correction on !" )
            #     self.flag = not self.flag
            # # time.sleep(0.2)
            # # self.mouse.position = new_cursor_position

            self.k += 1

            # print("dx : ", dx, " dy : ", dy, "dt : ", dt)
            # print("input model {}, pred : {}".format(model_input, prediction))
            print("real cursor {}, pred cursor {}".format((x, y), new_cursor_position))

            print(f"onmove time elapsed : {time.time() - t_start}")
            self.flag = not self.flag


    def on_press(self, key):

        if key == keyboard.Key.esc:
            exit()
        if key == keyboard.Key.space:
            self.mouse.position = (0, 0)



if __name__ == "__main__":

    model_type = "ANN" # model to load
    experiment = "P0_C0" # experiment to load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_log_path = f"results/{experiment}/{model_type}/"


    print("----------- Starting control for experiment : {} with model : {} -----------".format(experiment, model_type))
    
    # load model
    model = torch.load(model_log_path + "model.pt", weights_only=False).to("cpu")
    model = torch.jit.script(model)

    # controller creation 
    corrector = Corrector(model, device)
    with mouse.Listener(on_move=corrector.on_move), keyboard.Listener(on_press=corrector.on_press) as listener:
        listener.join()

