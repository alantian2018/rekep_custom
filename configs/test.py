import os

import yaml
import tqdm
import omnigibson as og

from omnigibson.utils.ui_utils import choose_from_options
import imageio


def main(random_selection=False, headless=True, short_exec=False):
    """
    Generates a BEHAVIOR Task environment in an online fashion.

    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)
 
    should_sample =False

    # Load the pre-selected configuration and set the online_sampling flag
    config_filename ='test.yaml'
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["task"]["online_object_sampling"] = should_sample

    # Load the environment
    assert gm.HEADLESS
    env = og.Environment(configs=cfg)
    print('Done with making ENV')
    video_writer = imageio.get_writer('./', fps=30)
    

    # Run a simple loop and reset periodically
    max_iterations = 10 if not short_exec else 1
    for j in tqdm.tqdm(range(max_iterations)):
        og.log.info("Resetting environment")
        env.reset()
        video_writer.append_data(env.render())
        for i in range(10):
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                og.log.info("Episode finished after {} timesteps".format(i + 1))
                break

    # Always close the environment at the end
    og.clear()
    video_writer.close()


if __name__ == "__main__":
    main()
