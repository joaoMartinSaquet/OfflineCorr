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

DEBUG_ONMOVE_POSITION = True
DEBUG_CORRECTION_STEP = True

class Corrector(object):
    """Controller with the model"""

    def __init__(self, model, device):

        self.corrector = model.eval()
        self.device = device

        # initialization
        self.x_poses = []
        self.y_poses = []
        self.do_correction = True # we can modify the mouse position 
        self.dxs = [0.]
        self.dys = [0.]
        self.t = time.time()
        self.k = 0
        self.correct = True
        
        # get the mouse
        self.mouse = mouse.Controller()
        self.x_poses.append(self.mouse.position[0])
        self.y_poses.append(self.mouse.position[1])

        self.min_input = np.array([0, 0, -80, -80, 0])
        self.max_input = np.array([1920, 1080, 80, 80, 500])




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

        model_input = np.array([self.x_poses[-1], self.x_poses[-1], dx, dy, dt]) # should i take current x and y or last one ? 

        # scale input
        # model_input = self.scale_input(model_input)
        
        model_input = torch.from_numpy(model_input).float()


        # get the prediction
        with torch.inference_mode():
            prediction = self.corrector(model_input)
    
        
        dx = prediction[0]
        dy = prediction[1]
        
        # compute the predicted cursor position
        cursor_position = (self.x_poses[-1] + dx, self.y_poses[-1] + dy)
        
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

    def scale_model_input(self, x):

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

        # redo the correction if the flag is down
        if not self.do_correction and not has_corrected:
            self.do_correction = True






    def on_press(self, key):

        if key == keyboard.Key.esc:
            self.listener.stop()
            # exit()
        if key == keyboard.Key.space:
            self.mouse.position = (1, 0)
        if key == keyboard.Key.ctrl:
            self.correct = not self.correct



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
        corrector.set_listener(listener)
        listener.join()
        print("----------- Control for experiment : {} with model : {} finished -----------".format(experiment, model_type))
        listener.stop()

