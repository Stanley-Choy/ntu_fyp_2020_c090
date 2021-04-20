import os
import gym
import gym_donkeycar
import numpy as np

# use the path to the simulator executable
host = 'localhost'
exe_path = f"C:/.../DonkeySimWin/donkey_sim.exe"
port = 9091 

conf = { "host": host, "exe_path" : exe_path, "port" : port }

env = gym.make("donkey-generated-roads-v0", conf=conf)

env.reset()

# drive forward at half throttle for 50 steps
for t in range(50):
    action = np.array([0,0.5])
    obv, reward, done, info = env.step(action)

    # view reward, cte, feedback
    #print(reward)
    #print(info['cte'])
    #print(obv)

env.close()
    