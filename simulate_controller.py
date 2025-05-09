from pynput import mouse, keyboard
import numpy as np
import matplotlib.pyplot as plt
import torch
import time



import os
import sys
file_path = os.path.abspath(__file__)
script_directory = os.path.dirname(file_path)
print("script directory : ", script_directory)
sys.path.append(os.path.abspath(script_directory))
sys.path.append("/home/jmartinsaquet/Documents/code/IA2_codes/OfflineCorr/utils")
from utils.model_handling import *
from collections import deque



'''
This script is a script that is going to load the trained model 
    - ANN
    - LSTM
    - CGP
    - GRN

and be tested on the user directly to see if there is any differences between with and without the controller
'''

DEBUG_ONMOVE_POSITION = True
DEBUG_CORRECTION_STEP = True

class Corrector(object):
    """Controller with the model"""

    def __init__(self, model, device, seq_length=None):

        self.corrector = model.eval()
        self.device = device
        self.seq_length = seq_length

        # initialization
        self.x_poses = []
        self.y_poses = []
        self.do_correction = True # we can modify the mouse position 
        self.dxs = [0.]
        self.dys = [0.]
        self.t = time.time()
        self.k = 0
        self.correct = False
        
        # get the mouse
        self.mouse = mouse.Controller()
        self.x_poses.append(self.mouse.position[0])
        self.y_poses.append(self.mouse.position[1])

        # with dt
        # self.min_input = np.array([0, 0, -80, -80, 0])
        # self.max_input = np.array([1920, 1080, 80, 80, 500])

        # without dt
        self.max_input = np.array([80, 80, 1000, 3.14])
        self.min_input = np.array([-80, -80, 0, 0])

        if seq_length is not None:
            self.model_input = deque([], maxlen=seq_length)

            for i in range(seq_length):
                self.model_input.append(np.zeros(4))

        print("Controller ready to be used !")
        print("Correction activated ? ", self.correct)

    def correct_displacement(self, x, y):
        # x and y are the cursor position after displacement

        # delta t wiht last dispalcement
        dt = time.time() - self.t 
        
        # get env_input
        dx = x - self.x_poses[-1]
        dy = y - self.y_poses[-1]

        # keep dx and dy for log purpose
        # self.dxs.append(dx)
        # self.dys.append(dy)
        if self.seq_length is None:
            model_input = np.array([self.x_poses[-1], self.x_poses[-1], dx, dy]) # should i take current x and y or last one ? 
            model_input = self.scale_input(model_input)

        else:
            # model_input = np.array([self.x_poses[-1], self.x_poses[-1], dx, dy]) # should i take current x and y or last one ? 
            model_input = np.array([dx, dy, dt, np.atan2(dy, dx)])
            # scale input
            model_input = self.scale_input(model_input)
            self.model_input.append(model_input)
            model_input = np.array(list(self.model_input))    
        

        model_input = torch.from_numpy(model_input).float()
        
        # get the prediction
        with torch.inference_mode():
            prediction = self.corrector(model_input.unsqueeze(0))[0]
    
        
        dx = prediction[0]
        dy = prediction[1]
        
        # compute the predicted cursor position
        cursor_position = (torch.round(self.x_poses[-1] + dx) , torch.round(self.y_poses[-1] + dy))
        
        print("x, y : ", (x,y), "cursor_position : ", cursor_position)
        
        # move the cursor position to the predicted position
        self.mouse.position = cursor_position

        # update of time
        self.t = time.time()

        self.x_poses.append(self.mouse.position[0])
        self.y_poses.append(self.mouse.position[1])
        
        # return correction has been done 
        return True
    

    def set_listener(self, listener):

        self.listener = listener

    def main(self):

        self.i = 0
        # do it still forever
        while True:
            pass
            # print("self.i ", self.i)
            # if self.i == 1000:
            #     print("move ! ")
            #     mouse.position = (0, 0)
            # self.i += 1
            # self.mouse.move(10,20)

                # self.flag = False

    def scale_input(self, x):

        return (x - self.min_input) / (self.max_input - self.min_input) * 2 - 1 

    def on_move(self, x, y):
        
        has_corrected = False

        if DEBUG_ONMOVE_POSITION:
            print("x : ", x, " y : ", y, "displacement_number : ", self.k)
            print("correct ? : ", self.correct, "do correction ?", self.do_correction)

        if self.do_correction and self.correct:
            has_corrected = self.correct_displacement(x, y)
            # if it has corrected the movement then we need to deactivate the correction for one turn (mouse.position = (0, 0) trigger callback)
            # print("we are after the correction step (i think we don't reach that step ! )")
            if has_corrected:
                self.do_correction = False

        self.k += 1

        # redue the correction if the flag is down
        if not self.do_correction and not has_corrected:
            self.do_correction = True

    def on_press(self, key):

        if key == keyboard.Key.esc:
            self.listener.stop()
            # exit()
        if key == keyboard.Key.space:
            self.mouse.position = (0, 0)

        if key == keyboard.Key.right:
            self.mouse.move(10,0)

        if key == keyboard.Key.left:
            self.mouse.move(-10,0)
        
        if key == keyboard.Key.down:
            self.mouse.move(0,10)
        
        if key == keyboard.Key.up:
            self.mouse.move(0,-10)
        
        if key == keyboard.Key.ctrl:
            self.correct = not self.correct



if __name__ == "__main__":

    model_type = "LSTM" # model to load
    experiment = "P0_C0" # experiment to load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_log_path = f"results/{experiment}/{model_type}/"
    
    # read hyperparameters
    config = load_config("config/ann_config.yaml")
    hyperparameters = config['hyperparameters']
    seq_length = hyperparameters['sequence_length']

    print("----------- Starting control for experiment : {} with model : {} -----------".format(experiment, model_type))
    
    # load model
    model = torch.load(model_log_path + "model.pt", weights_only=False).to("cpu")
    # model = torch.load(model_log_path + "model.pt", map_location=torch.device('cpu'))

    model = torch.jit.script(model)

    # controller creation 
    if model_type == "ANN":
        corrector = Corrector(model, device)
    elif model_type == "LSTM": 
        corrector = Corrector(model, device, seq_length)
    # elif model_type == "CGP":
    #     corrector = Corrector(model, device, seq_length)
    # elif model_type == "GRN":
    #     corrector = Corrector(model, device, seq_length)
    with mouse.Listener(on_move=corrector.on_move), keyboard.Listener(on_press=corrector.on_press) as listener:
        corrector.set_listener(listener)
        listener.join()
        print("----------- Control for experiment : {} with model : {} finished -----------".format(experiment, model_type))
        listener.stop()

