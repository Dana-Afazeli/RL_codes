import base64

import numpy as np
import matplotlib.pyplot as plt

from gym.wrappers.monitoring import video_recorder
from IPython.display import HTML
from IPython import display
import glob
import io

def decay_schedule(init_value, min_value, decay_ratio, episodes, log_start=-2, log_base=10):
    """
    used to create a list of exponentially decaying values in a certain number of steps
    :param init_value: float. init value
    :param min_value: float. minimum value
    :param decay_ratio: float. indicates the ratio of total steps in which value should be decayed
                            (after that, the minimum value gets repeated)
    :param episodes: int. total number of episodes. this will be the len of final list.
    :param log_start: float. power of the exponentially decaying values. Not a very important param
    :param log_base: float. base of the exponentially decaying values. Not a very important param
    :return: list(float). list of exponentially decaying values starting from index 0.
    """
    decay_steps = int(episodes * decay_ratio)
    rem_steps = episodes - decay_steps

    values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    values = np.pad(values, (0, rem_steps), 'edge')

    return values


def generate_trajectory(agent, env, max_steps=-1):
    """
    used to create a single rollout using the agent and environment given.
        you can specify max steps for the trajectories.
    :param agent: Agent. The agent to create rollouts with.
    :param env: Environment. The environment to create rollouts with.
    :param max_steps: int. optional. Maximum number of steps.
                the minimum number between this param and env.max_steps will be used.
    :return: List(tuple). Single trajectory as a list of (state, action, reward, next_state, done) tuples.
    """
    max_steps = min(max_steps, env.get_max_steps())

    step = 0
    done = False
    state = env.reset()
    trajectory = []
    while (max_steps == -1 or step < max_steps) and not done:
        action = agent.pi(state)
        next_state, reward, done, info = env.step(action)

        trajectory.append((state, action, reward, next_state, done))

        state = next_state
        step += 1

    return trajectory


def plot_curve(x, y, title, x_label, y_label, path=None, color='C0', window_size=None, lines=None):
    """
    util function that plots. that is it
    :param x: List.
    :param y: List.
    :param title: str. Title of the plot.
    :param x_label: str. Label of the x-axis.
    :param y_label: str. Label of the y-axis.
    :param path: str. Path at which the plot will be saved. if None, the plot will only be shown.
    :param color: str. Color of the data-points.
    :param window_size: int. Window to take the average on. if None, no averaging will be applied.
    :param lines: IDK
    :return: None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")

    if window_size is not None:
        n = len(y)
        final_y = np.empty(n)
        for t in range(n):
            final_y[t] = np.mean(y[max(0, t - window_size):(t + 1)])
    else:
        final_y = y

    ax.plot(x, final_y, color=color)
    ax.set_xlabel(x_label, color=color)
    ax.set_ylabel(y_label, color=color)
    ax.tick_params(axis='x', colors="k")
    ax.tick_params(axis='y', colors="k")
    ax.set_title(title)

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    if path is not None:
        plt.savefig(path)

    plt.close()

# These don't work yet!
def show_video(env_name):
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = 'video/{}.mp4'.format(env_name)
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")


def show_video_of_model(agent, env, video_path):
    vid = video_recorder.VideoRecorder(env, path=video_path)
    state = env.reset()
    done = False
    while not done:
        env.render(mode='rgb_array')
        vid.capture_frame()

        action = agent.pi(state)

        state, reward, done, _ = env.step(action)
    env.close()
