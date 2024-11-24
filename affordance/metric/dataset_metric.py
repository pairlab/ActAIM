import numpy as np
from scipy import ndimage
import torch.utils.data
from pathlib import Path
import os
import random
import pandas as pd
import pdb
from scipy.spatial.distance import cdist


class DatasetMetric(torch.utils.data.Dataset):
    def __init__(self, obs, batch_size):
        self.root = "./" + obs + "_dataset"
        self.object_list = os.listdir(self.root)
        self.num_object = len(self.object_list)
        self.batch_size = batch_size
        self.length = 2000

    def __len__(self):
        # think about what is the length of this dataset_single
        return self.length

    def __getitem__(self, index):
        dir_index = random.randint(0, self.num_object - 1)
        object_id = self.object_list[dir_index]
        object_dir = self.root + "/" + object_id
        dof_npz_path = object_dir + "/" + "dof.npz"
        # dof_npz_path = Path(dof_npz_path)
        dof = np.load(dof_npz_path)["dof"]
        dof_range = dof.shape[0]

        obs_array = None
        dof_array = np.zeros((self.batch_size, dof.shape[1]))

        for i in range(self.batch_size):
            index_sample = random.randint(0, dof_range - 1)
            obs_sample = self.get_obs(object_dir, index_sample)
            if i == 0:
                obs_array = np.zeros_like(obs_sample)
                obs_array = np.repeat(obs_array[np.newaxis, :], self.batch_size, axis=0)
            dof_array[i] = dof[index_sample]
            obs_array[i] = obs_sample

        dof_dist = cdist(dof_array, dof_array)
        return obs_array, dof_dist

        """
        index_x = random.randint(0, dof_range - 1)
        index_y = random.randint(0, dof_range - 1)

        dof_x = dof[index_x]
        dof_y = dof[index_y]

        object_obs_x_file = object_dir + "/" + str(index_x) + ".npz"
        object_obs_y_file = object_dir + "/" + str(index_y) + ".npz"

        object_obs_x = np.load(object_obs_x_file)['grid']
        object_obs_y = np.load(object_obs_y_file)['grid']

        return object_obs_x, object_obs_y, dof_x, dof_y
        """

    def get_obs(self, object_dir, index):
        object_obs_file = object_dir + "/" + str(index) + ".npz"
        object_obs = np.load(object_obs_file)["grid"]
        return object_obs


class DatasetModesMetric(torch.utils.data.Dataset):
    def __init__(self, obs, batch_size):
        self.root = "./modes_" + obs + "_dataset"
        self.object_list = os.listdir(self.root)
        self.num_object = len(self.object_list)
        # when collecting data
        self.num_envs = 50
        self.batch_size = batch_size
        self.length = 5000

        self.object_init_num = {}
        # maintain a dict to store number of initial state for each object
        for object_id in self.object_list:
            object_dir = self.root + "/" + object_id
            object_file_num = len(os.listdir(object_dir)) // (self.num_envs + 2)
            self.object_init_num[object_id] = object_file_num

    def __len__(self):
        # think about what is the length of this dataset_single
        return self.length

    def __getitem__(self, index):
        dir_index = random.randint(0, self.num_object - 1)
        object_id = self.object_list[dir_index]
        object_dir = self.root + "/" + object_id

        object_init_num = self.object_init_num[object_id]
        if object_init_num < 2:
            init_index = 0
        else:
            init_index = random.randint(0, object_init_num - 1)

        dof_npz_path = object_dir + "/" + str(init_index) + "_dof.npz"
        # dof_npz_path = Path(dof_npz_path)
        dof = np.load(dof_npz_path)["dof"]
        dof_range = dof.shape[0]

        obs_array = None
        dof_array = np.zeros((self.batch_size, dof.shape[1]))

        for i in range(self.batch_size):
            index_sample = random.randint(0, dof_range - 1)
            obs_sample = self.get_obs(object_dir, init_index, index_sample)
            if i == 0:
                obs_array = np.zeros_like(obs_sample)
                obs_array = np.repeat(obs_array[np.newaxis, :], self.batch_size, axis=0)
            dof_array[i] = dof[index_sample]
            obs_array[i] = obs_sample

        # dof_dist = cdist(dof_array, dof_array)
        dof_dist = np.sum(dof_array, axis=-1)

        init_obs_file = object_dir + "/init_" + str(init_index) + ".npz"
        init_obs = np.load(init_obs_file)["grid"]

        return obs_array, np.absolute(dof_dist), init_obs

    def get_obs(self, object_dir, init_index, index):
        object_obs_file = object_dir + "/" + str(init_index) + "_" + str(index) + ".npz"
        object_obs = np.load(object_obs_file)["grid"]
        return object_obs
