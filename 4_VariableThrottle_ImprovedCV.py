# adapted from the following sources
# https://github.com/flyyufelix/donkey_rl
# https://github.com/naokishibuya/car-finding-lane-lines

import sys
import gym
import gym_donkeycar
import random
import numpy as np
import cv2
import skimage as skimage
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras.initializers import identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
from keras import backend as K
from datetime import datetime
from matplotlib import pyplot as plt

import my_cv

# path to simulator executable
SIM_PATH = "C:/.../DonkeySimWin/donkey_sim.exe"
# path to save camera images for checking
IM_PATH = "C:/..."

# number of episodes
EPISODES = 2
# resize image to (80, 80, 4)
img_rows, img_cols = 80, 80
# convert image into b&w and stack 4 frames
img_channels = 4
# throttle value
STEERING = [-1,	-0.857142857,	-0.714285714,	-0.571428571,	-0.428571429,
                -0.285714286,	-0.142857143,	0,	0.142857143,	0.285714286,
                0.428571429,	0.571428571,	0.714285714,	0.857142857,	1]

THROTTLE = [0.1, 0.2, 0.2, 0.3, 0.3,
                0.3, 0.4, 0.5, 0.4, 0.3,
                0.3, 0.3, 0.2, 0.2, 0.1]
# enable lane detection 
LANE_DETECT = True
# memory length
DEQUE_LEN = 10000

global steering_index

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.t = 0
        self.max_Q = 0

        #| train | cont_ |
        #|   0   |   0   | no training, load saved model
        #|   0   |   1   | invalid
        #|   1   |   0   | train from the start, no loading
        #|   1   |   1   | load saved model and continue training

        self.train = False
        self.cont_train = False
        self.lane_detection = LANE_DETECT

        # obtain size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # DQN hyperparameters
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        # continue training from existing model
        if (self.train and self.cont_train):
            self.epsilon = 0.02
            self.initial_epsilon = 0.02
        # error checking
        elif (self.train==False and self.cont_train==True):
            raise ValueError("Check self.train and self.cont_train")
        # train a new model
        elif (self.train):
            self.epsilon = 1.0
            self.initial_epsilon = 1.0
        # load a trained model
        else:
            self.epsilon = 1e-6
            self.initial_epsilon = 1e-6
        self.epsilon_min = 0.02
        self.batch_size = 64
        self.train_start = 100
        self.explore = 10000

        # create replay memory using deque
        self.memory = deque(maxlen=DEQUE_LEN)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # copy model to target model
        # initialize the target model so that parameters are the same
        self.update_target_model()

    def build_model(self):
        print("Now we build the model")
        model = Sequential()
        model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="same",input_shape=(img_rows,img_cols,img_channels)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))

        # 15 output nodes for the 15 steering angles
        model.add(Dense(15, activation="linear")) 

        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse',optimizer=adam)
        print(model.summary()) 
        print("We finished building the model")

        return model
    
    def process_image(self, obs, e, im_count):
        # when lane_detection is false
        if not agent.lane_detection:
            # resize image and grayscale
            obs = skimage.color.rgb2gray(obs)
            obs = skimage.transform.resize(obs, (img_rows, img_cols))

            # view resized output
            #cv2.imshow("obs", obs)
            #cv2.waitKey(0)

            # save output as a jpg
            #file_name = "/" + str(LANE_DETECT) + "_" + str(THROTTLE) + "_" + str(e+1) + "_" + str(im_count) + ".jpg"
            ##print(file_name)
            #cv2.imwrite(IM_PATH+file_name, obs*255)

            return obs
        else:
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs = cv2.resize(obs, (img_rows, img_cols))
            obs = cv2.GaussianBlur(obs, (5, 5), 0)

            #cv2.imshow("obs", obs)
            #cv2.waitKey(0)

            #file_name_a = "/" + str(LANE_DETECT) + "_" + str(THROTTLE) + "_" + str(e+1) + "_" + str(im_count) + "_a" + ".jpg"
            #print(file_name)
            #cv2.imwrite(IM_PATH+file_name_a, obs)

            low_threshold = 50
            high_threshold = 150
            edges = my_cv.detect_edges(obs, low_threshold=50, high_threshold=150)
            #cv2.imshow("edges", edges)
            #cv2.waitKey(0)

            #file_name_b = "/" + str(LANE_DETECT) + "_" + str(THROTTLE) + "_" + str(e+1) + "_" + str(im_count) + "_b" + ".jpg"
            ##print(file_name)
            #cv2.imwrite(IM_PATH+file_name_b, edges)

            ret_img = edges

            return ret_img
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # obtain action from model using e-greedy policy
    def get_action(self, s_t):
        if np.random.rand() <= self.epsilon:
            #print("Return Random Value")
            #return random.randrange(self.action_size)
                        
            global steering_index
            steering_index = random.randint(0, 14)
            #print("here")
            return STEERING[steering_index]
        else:
            #print("Return Max Q Prediction")
            q_value = self.model.predict(s_t)
            # Convert q array to steering value
            #print("there")
            return linear_unbin(q_value[0])

    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            #self.epsilon *= self.epsilon_decay
            self.epsilon -= (self.initial_epsilon - self.epsilon_min) / self.explore

    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
        state_t = np.concatenate(state_t)
        state_t1 = np.concatenate(state_t1)
        targets = self.model.predict(state_t)
        self.max_Q = np.max(targets[0])
        target_val = self.model.predict(state_t1)
        target_val_ = self.target_model.predict(state_t1)
        for i in range(batch_size):
            if terminal[i]:
                targets[i][action_t[i]] = reward_t[i]
            else:
                a = np.argmax(target_val[i])
                targets[i][action_t[i]] = reward_t[i] + self.discount_factor * (target_val_[i][a])

        self.model.train_on_batch(state_t, targets)

    def load_model(self, name):
        self.model.load_weights(name)

    # save the currently training model
    def save_model(self, name):
        self.model.save_weights(name)

# converts a number into corresponding the 15-bit array
def linear_bin(a):
    a = a + 1
    b = round(a / (2 / 14))

    arr = np.zeros(15)
    arr[int(b)] = 1
    return arr

# converts the array into a floating point number
def linear_unbin(arr):
    if not len(arr) == 15:
        raise ValueError('Illegal array length, must be 15')
    b = np.argmax(arr)
    global steering_index
    steering_index = b

    a = b * (2 / 14) - 1
    return a

if __name__ == "__main__":

    # prints the starting time into a text file
    start_time = datetime.now()
    start_time_print = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file = open("time.txt", "w")
    file.write("Started:\t%s\n" %start_time_print)
    file.close()

    # configurations to run the Donkey Car simulator
    host = "localhost"
    exe_path = SIM_PATH
    port = 9091 
    conf = { "host": host, "exe_path" : exe_path, "port" : port }
    track = "donkey-generated-roads-v0"
    #track = "donkey-generated-track-v0"
    #track = "donkey-warehouse-v0"

    # build the environment
    env = gym.make(track, conf=conf)

    # get size of state and action from environment
    state_size = (img_rows, img_cols, img_channels) # 80, 80, 4
    action_size = env.action_space.shape[0] # 2

    # calls the class
    agent = DQNAgent(state_size, action_size)

    episodes = []
    episode_length_memory = []
    episode_reward_memory = []
    episode_reward_movavg_memory = []

    if agent.train and agent.cont_train:
        print("\n---\nTraining will now resume from saved model.")
    elif agent.train:
        print("\n---\nTraining will now start.")
    else:
        print("\n---\nThe saved model will be loaded.")
        agent.load_model("save_model.h5")

    for e in range(EPISODES):
        print("---")
        print("Episode:", e+1)
        done = False
        obs = env.reset()
        episode_len = 0
        episode_reward = 0.0
        im_count = 1
       
        x_t = agent.process_image(obs, e, im_count)
        im_count += 1
        s_t = np.stack((x_t,x_t,x_t,x_t),axis=2)
        s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2]) #1*80*80*4       
        
                
        while not done:
            # action for current state
            steering = agent.get_action(s_t)
            throttle = THROTTLE[steering_index]
           
            action = [steering, throttle]
            #print(steering_index, steering, throttle, sep="\t")

            # advance one step
            next_obs, reward, done, info = env.step(action)
            #print(info)

            x_t1 = agent.process_image(next_obs, e, im_count)
            im_count += 1
            x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
            s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3) #1x80x80x4

            # save the sample <s, a, r, s'> to the replay memory
            agent.replay_memory(s_t, np.argmax(linear_bin(steering)), reward, s_t1, done)

            if agent.train:
                agent.train_replay()

            s_t = s_t1
            agent.t = agent.t + 1
            episode_len = episode_len + 1
            episode_reward = episode_reward + reward

            #if agent.t % 30 == 0:
                #print("EPISODE",  e, "TIMESTEP", agent.t,"/ ACTION", action, "/ REWARD", reward, "/ EPISODE LENGTH", episode_len, "/ Q_MAX " , agent.max_Q)

            # cut off at 1000 episodes for longer runs
            if episode_len >= 1000:
                done = True

            if done:
                # update the target model to be same with model
                agent.update_target_model()
                episodes.append(e)
                episode_length_memory.append(episode_len)
                episode_reward_memory.append(episode_reward)
                episode_reward_movavg_memory.append(np.mean(episode_reward_memory))
                # save model for each episode
                if agent.train:
                    agent.save_model("save_model.h5")

                print("Episode: %d\tMemory Length: %d\tEpsilon: %.5f\tEpisode Length: %d\tEpisode Reward: %.2f" 
                      % (e+1, len(agent.memory), agent.epsilon, episode_len, episode_reward))
    
    #env.close()
    end_time = datetime.now()
    end_time_print = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration = end_time - start_time
    duration_print = str(duration).split(".")[0]
    mean_reward = np.mean(episode_reward_memory)
    mean_length = np.mean(episode_length_memory)

    # writes the ending time and duration to the text file as well as the necessary results
    file = open("time.txt", "a")
    file.write("Ended:\t\t%s\n" % end_time_print)
    file.write("Duration:\t%s\n" % duration_print)
    file.write("Episodes:\t%d\n" % (e+1))
    file.write("Mean Reward:\t%.5f\n" % (mean_reward))
    file.write("Mean Length:\t%d\n" % (mean_length))
    #file.write("Throttle:\t%.1f\n" % (THROTTLE))
    file.write("Lane Detection:\t%s\n" % (LANE_DETECT))    
    
    if (agent.train and agent.cont_train):
        file.write("\nTraining was continued from a saved model\n")
    elif (agent.train):
        file.write("\nTraining was newly conducted\n")
    else:
        file.write("\nA saved model was loaded. No training was conducted.\n")

    file.write("\n" + str(episode_length_memory) + "\n")
    file.write("\n" + str(episode_reward_memory) + "\n")
    file.write("\n" + str(episode_reward_movavg_memory) + "\n")

    file.close()