#!/usr/bin/env python

import click
import numpy as np
import gym
import matplotlib.pyplot as plt

def test_get_action(theta, ob, rng=np.random):
    ob_1 = np.hstack((ob,1))    
    mean = theta.dot(ob_1)
    return mean

def chakra_get_action(theta, ob, rng=np.random):
    ob_1 = np.hstack((ob,1))    
    mean = theta.dot(ob_1)
    return rng.normal(loc=mean, scale=1.)		 

@click.command()
@click.argument("env_id", type=str, default="vishamC")
def main(env_id):
    # Register the environment
    rng = np.random.RandomState(42)

    if env_id == 'vishamC':
        from rlpa2 import vishamC
        env = gym.make('vishamC-v0')
        get_action = chakra_get_action
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    else:
        raise ValueError(
            "Unsupported environment: must be 'chakra' ")

    env.seed(42)

    # Initialize parameters
    theta = rng.normal(scale=0.01, size=(action_dim, obs_dim + 1))
    bs_theta = np.random.rand(3,1)
    alpha = 0.01#0.0005
    alpha_bs = 0.05#0.0005
    gamma = 0.9
    max_itr = 60
    batch_size = 20
    T = 200        
    
    for itr in range(max_itr):

        grad = 0
        grad_bs = 0
        
        for n_samples in range(batch_size):
            ob = env.reset()
            ob_1 = np.hstack((ob,1))
            ob_1 = ob_1.reshape(3,1)
            done = False    

            # Only render the first trajectory
            # Collect a new trajectory            
            states = [ob_1]
            actions = []
            rewards = []

            R_t = np.zeros(T)
            
            for t in range(T):
                action = get_action(theta, ob, rng=rng)            
                next_ob, rew, done, _ = env.step(action)
                ob = next_ob
                ob_1 = np.hstack((ob,1))                
                ob_1 = ob_1.reshape(3,1)                
                action = action.reshape(2,1)
                if n_samples == -1:
                    env.render(mode='human')
                actions.append(action)
                rewards.append(rew)
                states.append(ob_1)
            
            for t in range(T):
                for tprime in range(t,T):
                    R_t[t] += (gamma**(tprime - t)) * rewards[tprime]
            
            for t in range(T):
                
                a = actions[t] - theta.dot(states[t])
                
                del_log_p = (a).dot((states[t]).transpose())
                
                Advantage = R_t[t] - (bs_theta.transpose()).dot(states[t]**2)

                grad_tmp = del_log_p * (Advantage)

                grad += grad_tmp / (np.linalg.norm(grad_tmp) + np.exp(-8))
                
                grad_bs_tmp = (states[t]**2)*Advantage

                grad_bs = grad_bs_tmp / (np.linalg.norm(grad_bs_tmp)+ np.exp(-8))

                bs_theta += alpha_bs * grad_bs

            theta += alpha * grad/batch_size
            
        
        print(itr)
        print(np.mean(rewards))
        print(theta)
        #print(bs_theta)

    x_axis = np.arange(-1,1,0.1)
    y_axis = x_axis
    x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)
    direction_x = []
    direction_y = []

    for j1 in np.arange(-1,1,0.1):
        tmp_x = []
        tmp_y = []
        for j2 in np.arange(-1,1,0.1):            
            point_dir = theta.dot(np.array([j1, j2, 1]))
            print(point_dir[0]**2 + point_dir[0]**2)
            tmp_x.append(point_dir[0])
            tmp_y.append(point_dir[1])
        direction_x.append(tmp_y)
        direction_y.append(tmp_x)

    plt.quiver(y_mesh, x_mesh, np.array(direction_y), np.array(direction_x))
    plt.title('Policy trajectories')
    plt.savefig('PG_1.png')
    plt.show()

    contr = []
    
    for j1 in range(len(x_axis)):
        contr_point = []
        for j2 in range(len(y_axis)):
            point_tmp = (np.transpose(bs_theta)).dot(np.square([x_axis[j1], y_axis[j2], 1]))
            contr_point.append(point_tmp.item())
        contr.append(contr_point)

    plt.figure()
    plt.contour(y_mesh, x_mesh, np.array(contr))
    plt.title('Value function visualization')
    plt.savefig('PG_2.png')
    plt.show()
        
    testing = 0
    while True:        
        action = test_get_action(theta, env.state, rng=rng)            
        next_ob, rew, done, _ = env.step(action)    
        env.render(mode='human')
        testing += 1
        if testing % 100 == 0:
            env.reset()

                        
if __name__ == "__main__":
    main()
