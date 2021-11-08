import os
import gym
import abc
import glob
import numpy as np
import matplotlib.pyplot as plt 
# import cv2

# import kornia
import torchvision

import icedata
from icevision.all import BBox, show_sample, draw_sample, draw_pred, draw_bbox
from icevision.models.inference import draw_img_and_boxes
from icevision.utils import denormalize_imagenet 
from argparse import ArgumentParser
from PIL import Image
from pprint import pprint
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
# from stable_baselines.common.vec_env import DummyVecEnv

class BaseCustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    # reward_range = (0.0, 1.0)
    @abc.abstractmethod
    def __init__(self):
        self.__version__ = "0.0.1"
        print("Init CustomEnv")
        # Modify the observation space, low, high and shape values according to 
        # your custom environment's needs
        # self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,))
        # Modify the action space, and dimension according to your custom environment's needs
        # self.action_space = gym.spaces.Discrete(4)
        pass

    @abc.abstractmethod
    def step(self, action):
        """
        Runs one time-step of the environment's dynamics. 
        The reset() method is called at the end of every episode
        :param action: The action to be executed in the environment
        :return: (observation, reward, done, info)
            observation (object):
                Observation from the environment at the current time-step
            reward (float):
                Reward from the environment due to the previous action performed
            done (bool):
                a boolean, indicating whether the episode has ended
            info (dict):
                a dictionary containing additional information about the previous action
        """
        # Implement your step method here
        # return (observation, reward, done, info)
        pass

    @abc.abstractmethod
    def reset(self):
        """
        Reset the environment state and returns an initial observation
        Returns
        -------
        observation (object): The initial observation for the new episode after reset
        :return:
        """
        # Implement your reset method here
        # return observation
        pass

    @abc.abstractmethod
    def render(self, mode='human', close=False):
        """
        :param mode:
        :return:
        """
        pass


from icevision.imports import *
from icevision.core import *
from icevision.data import *
# from icevision.tfms.albumentations.albumentations_helpers import (
#     get_size_without_padding,
# )
# from icevision.tfms.albumentations import albumentations_adapter

from icevision.utils.imageio import *
# from icevision.visualize.draw_data import *
# from icevision.visualize.utils import *


class BoundingBoxEnv(BaseCustomEnv):
    @property
    def observation_space(self):
        return gym.spaces.Box(low=0, high=np.inf, shape=(3, self.shape, self.shape), dtype=np.float32)

    @property
    def action_space(self):
        return gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

    def __init__(self, shape=384, episode_length=100):
        self.data_dir = icedata.fridge.load_data()
        self.class_map = icedata.fridge.class_map() 
        self.train_ds, self.valid_ds = icedata.fridge.dataset(self.data_dir)
        self.shape = shape  
        self.width = shape
        self.height = shape
        self.episode_length = episode_length
        
        self.reset()
    
    def reset(self):
        dataset = self.valid_ds
        random_index = np.random.randint(len(dataset))
        self.sample = dataset[random_index]
        self.screen = ((1*self.sample.img)).astype(np.uint8)
        self.current_step = 0

        observation = torchvision.transforms.ToTensor()(self.sample.img)

        self.bboxes = []
        self.videos = []
        return observation

    def step(self, action):
        self.current_step += 1
        xcycwh = action #(0.7, 0.2, 0.1, 0.2)
        bbox = BBox.from_relative_xcycwh(*xcycwh, img_width=self.width, img_height=self.height)
        bbox.autofix(img_w=self.width, img_h=self.height)
        self.bboxes.append(bbox)

        next_state = torchvision.transforms.ToTensor()(self.sample.img)


        reward = 0
        done = (self.current_step >= self.episode_length)
        # if done:
        #     self.videos[0].save('vid.gif',
        #        save_all=True, 
        #        append_images=self.videos[1:], 
        #        optimize=False, 
        #        duration=10, 
        #        loop=0)
        info = {}
        return next_state, reward, done, info 

    def draw_img_and_boxes(self, 
        img: Union[PIL.Image.Image, np.ndarray],
        bboxes: dict,
        class_map,
        display_score: bool = True,
        label_color: Union[np.array, list, tuple, str] = (255, 255, 0),
        label_border_color: Union[np.array, list, tuple, str] = (255, 255, 0),
    ) -> PIL.Image.Image:

        if not isinstance(img, PIL.Image.Image):
            img = np.array(img)

        # convert dict to record
        record = ObjectDetectionRecord()
        w, h = img.shape[:2]
        record.img = np.array(img)
        record.set_img_size(ImgSize(width=w, height=h))
        record.detection.set_class_map(class_map)
        for bbox in bboxes:
            record.detection.add_bboxes(bboxes)
            # record.detection.add_labels([bbox["class"]])
            # record.detection.set_scores(bbox["score"])

        pred_img = draw_sample(
            record,
            display_score=display_score,
            label_color=label_color,
            label_border_color=label_border_color,
        )

        return pred_img
        
    def render(self, mode='human', close=False):
        if self.bboxes:
            pprint(self.bboxes[-1]) 
        vis = self.draw_img_and_boxes(
            denormalize_imagenet(self.sample.img),
            self.bboxes, 
            self.class_map, display_score=False
        )
        plt.imshow(vis)
        plt.axis("off")
        plt.pause(1)
        self.videos.append(Image.fromarray(vis))



if __name__ == '__main__':


    MAX_NUM_EPISODES = 1
    MAX_STEPS_PER_EPISODE = 20

    env = BoundingBoxEnv(episode_length=MAX_STEPS_PER_EPISODE)
    env = DummyVecEnv([lambda: env])


    for episode in range(MAX_NUM_EPISODES):
        observation = env.reset()
        for step in range(MAX_STEPS_PER_EPISODE):
            [env.render() for n in range(5)]
            action = [env.action_space.sample() for n in range(5)]# Sample random action. This will be replaced by our agent's action when we start developing the agentalgorithms
            next_state, reward, done, info = env.step(action) # Send the action to the environment and receive the next_state, reward and whether done or not
            observation = next_state
            if done is True:
                print("\n Episode #{} ended in {} steps.".format(episode, step+1))
            # break