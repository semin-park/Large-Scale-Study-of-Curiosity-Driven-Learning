import torch, numpy as np
import matplotlib.pyplot as plt

def plot(step, score, int_reward, ploss, vloss, entropy, name):
    plt.figure(figsize=(20,10))

    plt.subplot(231)
    plt.title(f'Running avg. score: {np.mean(score[-10:]):.2f}, Frame: {step}')
    plt.xlabel('# episodes')
    plt.ylabel('score')
    plt.plot(score)
    
    plt.subplot(232)
    plt.title(f'intrinsic reward: {np.mean(int_reward[-1000:]):.4e} (mean of 1K)')
    plt.xlabel('# frames')
    plt.ylabel('reward')
    plt.plot(int_reward)

    # plt.subplot(235)
    # plt.title(f'extrinsic reward: {np.mean(ext_reward[-1000:]):.4f} (mean of 1K)')
    # plt.xlabel('# frames')
    # plt.ylabel('reward')
    # plt.plot(ext_reward)

    plt.subplot(233)
    plt.title(f'ploss: {np.mean(ploss[-1000:]):.4e} (mean of 1K)')
    plt.xlabel('# frames')
    plt.ylabel('ploss')
    plt.plot(ploss)

    plt.subplot(236)
    plt.title(f'vloss: {np.mean(vloss[-1000:]):.4e} (mean of 1K)')
    plt.xlabel('# frames')
    plt.ylabel('vloss')
    plt.plot(vloss)

    plt.subplot(234)
    plt.title(f'entropy: {np.mean(entropy[-1000:]):.4f} (mean of 1K)')
    plt.xlabel('# frames')
    plt.ylabel('entropy')
    plt.plot(entropy)

    plt.savefig(name)
    plt.close()