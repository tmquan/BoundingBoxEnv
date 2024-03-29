import os
import gym
from gym.utils import seeding

import abc
import glob
import random
import numpy as np
import matplotlib.pyplot as plt 
import cv2
# import warnings
# warnings.filterwarnings('ignore')
import albumentations as AB
from albumentations.pytorch import ToTensorV2

import kornia
import torchvision
from torchvision import transforms
import icedata
from icevision.all import tfms, BBox, show_sample, draw_sample, draw_pred, draw_bbox, draw_record
from icevision.models.inference import draw_img_and_boxes
from icevision.utils import denormalize_imagenet 
from argparse import ArgumentParser
from PIL import Image
from pprint import pprint
# from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
# from stable_baselines.common import set_global_seeds
# from stable_baselines.common.vec_env import DummyVecEnv

import skimage.io
import skimage.color
import skimage.measure
import skimage.segmentation

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss

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

    def __init__(self, shape=384, episode_length=100, is_train_unet=True, is_model_load=True):
        # self.data_dir = icedata.fridge.load_data()
        # self.class_map = icedata.fridge.class_map() 
        # # self.train_ds, self.valid_ds = icedata.fridge.dataset(self.data_dir)
        # parser = icedata.fridge.dataset(self.data_dir)
        
        # np.random.seed(123)
        
        self.data_dir = icedata.pennfudan.load_data()
        self.class_map = ClassMap(['person']) 
        parser = icedata.pennfudan.parser(self.data_dir)
        data_splitter = RandomSplitter([0.8, 0.2], seed=42)
        self.train_records, self.valid_records = parser.parse(data_splitter)


        shift_scale_rotate = tfms.A.ShiftScaleRotate(rotate_limit=10)
        crop_fn = partial(tfms.A.RandomSizedCrop, min_max_height=(384 // 2, 384 // 2), p=.3)
        train_tfms = tfms.A.Adapter(
            [
                # *tfms.A.aug_tfms(size=384, presize=512, 
                #     # shift_scale_rotate=shift_scale_rotate, 
                #     # crop_fn=crop_fn
                #     ),
                tfms.A.Normalize(),
            ]
        )
        valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=348), tfms.A.Normalize()])

        self.train_ds = Dataset(self.train_records, train_tfms)
        self.valid_ds = Dataset(self.valid_records, valid_tfms)
        # self.train_ds, self.valid_ds = icedata.pennfudan.dataset(self.data_dir)
        
        # self.shape = shape  
        # self.width = shape
        # self.height = shape
        self.episode_length = episode_length
        
        # self.reset() # Reset needs explicitly called to obtain observations
        self.unet = nn.Sequential(
            UNet(
                spatial_dims=2,
                in_channels=3,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=3,
                norm=Norm.BATCH,
            ), 
            nn.Sigmoid()
        )
        self.is_train_unet = is_train_unet
        self.is_model_load = is_model_load
        self.loss_function = DiceLoss()
        self.loss = 1.
        
        # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_unet_model_path = 'best_unet_model.pt'
        if self.is_model_load and os.path.exists(self.best_unet_model_path):
            print("Load from previous Mask UNet..")
            self.unet.load_state_dict(torch.load(self.best_unet_model_path))
        self.unet.to(self.device)
        self.learning_rate = 1e-3
        self.optimizer = torch.optim.RMSprop(self.unet.parameters(), 
                                             lr=self.learning_rate, 
                                             weight_decay=1e-8, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=2)  # goal: maximize Dice score


    def reset(self):
        dataset = self.train_ds
        random_index = np.random.randint(len(dataset))
        self.record = dataset[random_index]
        # pprint(self.record)

        self.current_step = 0

        # Process the instance
        # TODO
        # Groundtruth will have surfix underscore
        self._bboxes = self.record.detection.bboxes
        self._masks = self.record.detection.mask_array
        print(len(self._bboxes))
        print(self._masks.shape)
        self._mask = np.zeros(self._masks.shape[-2:], dtype=np.uint8)
        for num in range(self._masks.shape[0]):
            self._mask += (num+1) * np.array(self._masks.to_tensor().numpy()[num])
        # print(self._mask.dtype, self._mask.shape)
        self._image = self.record.img
        self._label = skimage.measure.label(self._mask)
        self._bndry = skimage.segmentation.find_boundaries(self._label)
        self._fgrnd = np.bitwise_and(np.bitwise_not(self._bndry), (self._label > 0))
        self._color = skimage.color.label2rgb(self._mask, bg_label=0)
        self.height, self.width = self._image.shape[:2]

        # Save for debugging
        skimage.io.imsave("_label.png", self._label)
        skimage.io.imsave("_bndry.png", self._bndry)
        skimage.io.imsave("_fgrnd.png", self._fgrnd)
        skimage.io.imsave("_color.png", self._color)

        # Form the observation, image will be a condition
        self.image = self.record.img
        self.bndry = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.bboxes = []
        self.videos = []

        # observation = torchvision.transforms.ToTensor()(self.record.img)
        observation = self._get_observation(self.image, self.bndry, self.bboxes)
        return observation

    def _get_observation(self, image, bndry, bboxes):
        image_info = image
        bndry_info = bndry
        # Process the bounding box
        bbox_info = np.zeros(image.shape[:2], dtype=np.uint8)
        for bbox in bboxes:
            (xmin, ymin, xmax, ymax) = bbox.xyxy
            (xmin, ymin, xmax, ymax) = (int(xmin), int(ymin), int(xmax), int(ymax))
            # print(xmin, ymin, xmax, ymax)
            bbox_info[ymin:ymax, xmin:xmax] = 255
        
        image_tensor = kornia.image_to_tensor(image_info)
        bndry_tensor = kornia.image_to_tensor(bndry_info)
        bbox_tensor = kornia.image_to_tensor(bbox_info)
        concat = torch.cat([image_tensor, bndry_tensor, bbox_tensor], dim=0)
        # print(image_tensor.shape)
        # print(bbox_tensor.shape)
        # print(bndry_tensor.shape)
        # print(concat.shape)
        # skimage.io.imsave("_s_image.png", image_info * bbox_info[...,None] / 255)
        # skimage.io.imsave("_s_bndry.png", bndry_info * bbox_info / 255)
        # skimage.io.imsave("_s_bbox.png", bbox_info)
        # cv2.imshow("BBox", bbox_info)
        # cv2.waitKey(1)
        plt.figure(2)
        plt.imshow(bbox_info)
        plt.axis("off")
        plt.pause(0.1)
        return concat

    def _train_unet(self, image, fgrnd, bbox):
        # Extract the cropped region
        xmin, ymin, xmax, ymax = bbox.xyxy

        # Crop both image and foreground
        tfms = AB.Compose([
            AB.Crop(x_min=xmin, y_min=ymin, x_max=xmax, y_max=ymax),
            AB.Resize(width=256, height=256),
            AB.pytorch.ToTensorV2(),
        ])
        # Convert to torch tensor
        augmented = tfms(image=image.astype(np.uint8), 
                         mask=fgrnd.astype(np.uint8),
                         mask_interpolation=cv2.INTER_NEAREST_EXACT)
        tensor_image = augmented["image"].to(self.device).unsqueeze_(0) / 255.0
        tensor_fgrnd = augmented["mask"].to(self.device).unsqueeze_(0).unsqueeze_(1) / 255.0
        
        # zero the parameter gradients
        self.optimizer.zero_grad()
        tensor_estim = self.unet.forward(tensor_image)
        # print(tensor_image.shape, tensor_estim.shape, tensor_fgrnd.shape)
        self.loss = self.loss_function(tensor_estim, tensor_fgrnd)
        self.loss.backward()
        self.optimizer.step()
        return self.loss.item()

    def step(self, action, is_train_unet=True):
        self.current_step += 1
        if False: # xywh
            xcycwh = action #(0.7, 0.2, 0.1, 0.2)
            bbox = BBox.from_relative_xcycwh(*xcycwh, 
                                             img_width=self.width, 
                                             img_height=self.height)
        else:
            xmin = int(round(action[0] * (self.width - 2)))
            ymin = int(round(action[1] * (self.height - 2)))
            xmax = int(round(action[2] * (self.width - xmin - 2))) + 1 + xmin
            ymax = int(round(action[3] * (self.height - ymin - 2))) + 1 + ymin
            bbox = BBox.from_xyxy(xmin, ymin, xmax, ymax)
        bbox.autofix(img_w=self.width, img_h=self.height)
        self.bboxes.append(bbox)

        # Train UNet for segmentation in the box
        if self.is_train_unet:
            self.unet.train() # Set the model to training mode
            loss = self._train_unet(self._image, self._fgrnd, bbox)
            print("Loss: ", loss)  
        else:
            # self.unet.eval()    
            pass 
        # next_state = torchvision.transforms.ToTensor()(self.record.img)
        next_state = self._get_observation(self.image, self.bndry, self.bboxes)

        reward = 0
        done = (self.current_step >= self.episode_length)
        # if done:
        #     self.videos[0].save('_video.gif',
        #        save_all=True, 
        #        append_images=self.videos[1:], 
        #        optimize=False, 
        #        duration=10, 
        #        loop=0)
        info = {}
        return next_state, reward, done, info 

    # def draw_img_and_boxes(self, 
    #     img: Union[PIL.Image.Image, np.ndarray],
    #     bboxes: dict, 
    #     class_map,
    #     display_score: bool = True,
    #     label_color: Union[np.array, list, tuple, str] = (255, 255, 0),
    #     label_border_color: Union[np.array, list, tuple, str] = (255, 255, 0),
    # ) -> PIL.Image.Image:

    #     if not isinstance(img, PIL.Image.Image):
    #         img = np.array(img)

    #     # convert dict to record
    #     record = ObjectDetectionRecord()
    #     record.add_component(InstanceMasksRecordComponent())
    #     # record = InstanceSegmentationRecord()
    #     w, h = img.shape[:2]
    #     record.img = np.array(img)
    #     record.set_img_size(ImgSize(width=w, height=h))
    #     record.detection.set_class_map(class_map)
    #     record.detection.add_bboxes(bboxes)
    #     record.detection.add_masks([self._masks])
    #     # for bbox in bboxes:
    #     #     record.detection.add_bboxes(bbox)
    #     #     record.detection.add_labels([bbox["class"]])
    #     #     record.detection.set_scores(bbox["score"])

    #     # pred_img = draw_sample(
    #     #     record,
    #     #     display_score=display_score,
    #     #     label_color=label_color,
    #     #     label_border_color=label_border_color,
    #     # )
    #     pred_img = draw_record(
    #         record=record,
    #         class_map=self.class_map,
    #         display_label=True,
    #         display_bbox=True,
    #         display_mask=True,
    #     )

    #     return pred_img
        
    def render(self, mode='human', close=False):
        if self.bboxes:
            pprint(self.bboxes[-1]) 
        # vis = self.draw_img_and_boxes(
        #     denormalize_imagenet(self.record.img),
        #     self.bboxes, 
        #     self.class_map, 
        #     display_score=False
        # )
        _vis = draw_record(
            record=self.record,
            class_map=self.class_map,
            display_label=False,
            display_bbox=True,
            display_mask=True,
        )
        plt.figure(1)
        plt.imshow(_vis)
        plt.axis("off")
        plt.pause(0.1)
        # self.videos.append(Image.fromarray(_vis))



if __name__ == '__main__':


    MAX_NUM_EPISODES = 1000
    MAX_STEPS_PER_EPISODE = 20

    env = BoundingBoxEnv(
        episode_length=MAX_STEPS_PER_EPISODE,
        is_train_unet=True,
        is_model_load=True
    )
    RANDOM_SEED = 2021
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)
    
    best_metric = -1
    best_metric_epoch = -1
    for episode in range(MAX_NUM_EPISODES):
        env.seed(np.random.randint(0, 2**32 - 1))
        observation = env.reset()
        env.render()
        for step in range(MAX_STEPS_PER_EPISODE):
            # env.render()
            # np.random.seed(step)
            action = np.random.uniform(0, 1, env.action_space.shape)
            # env.action_space.seed(RANDOM_SEED)
            # action = env.action_space.sample()
            next_state, reward, done, info = env.step(action, is_train_unet=True) # Send the action to the environment and receive the next_state, reward and whether done or not
            observation = next_state
            if done is True:
                print("\n Episode #{} ended in {} steps.".format(episode, step+1))
            # break

            metric = 1 - env.loss # DiceScore = 1 - DiceLoss
            if metric > best_metric:
                best_metric = metric
                # best_metric_epoch = epoch + 1
                best_metric_epoch = episode * MAX_STEPS_PER_EPISODE + step + 1
                torch.save(env.unet.state_dict(), os.path.join(os.getcwd(), env.best_unet_model_path))
                print("saved new best metric model: ", metric)