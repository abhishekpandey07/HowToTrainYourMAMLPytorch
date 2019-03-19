import json
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import tqdm
import concurrent.futures
import pickle
import torch
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from collections import defaultdict
from utils.parser_utils import get_args
from utils.tensor_generator import TensorGenerator
import string
from dataproviders.twitter_dataprovider import TwitterDataProvider
from dataproviders.pan_dataprovider import PANDataProvider


class DataProviderFactory(object):
    @classmethod
    def get_loader(self, name="twitter") -> Dataset:
        if("twitter" in name.lower()):
            print('using TwitterDataProvider')
            return TwitterDataProvider
        elif("pan" in name.lower()):
            print('using PanDataProvider')
            return PANDataProvider
        

class MetaLearningSystemDataLoader(object):
    def __init__(self, args, current_iter=0,):
        """
        Initializes a meta learning system dataloader. The data loader uses the Pytorch DataLoader class to parallelize
        batch sampling and preprocessing.
        :param args: An arguments NamedTuple containing all the required arguments.
        :param current_iter: Current iter of experiment. Is used to make sure the data loader continues where it left
        of previously.
        """
        self.num_of_gpus = args.num_of_gpus
        self.batch_size = args.batch_size
        self.samples_per_iter = args.samples_per_iter
        self.num_workers = args.num_dataprovider_workers
        self.total_train_iters_produced = 0
        self.dataset = DataProviderFactory.get_loader(name=args.dataset_name)(args=args)
        self.batches_per_iter = args.samples_per_iter
        self.full_data_length = self.dataset.data_length
        self.continue_from_iter(current_iter=current_iter)
        self.args = args

    def get_dataloader(self):
        """
        Returns a data loader with the correct set (train, val or test), continuing from the current iter.
        :return:
        """
        return DataLoader(self.dataset, batch_size=(self.num_of_gpus * self.batch_size * self.samples_per_iter),
                          shuffle=False, drop_last=True)#,num_workers=self.num_workers)

    def continue_from_iter(self, current_iter):
        """
        Makes sure the data provider is aware of where we are in terms of training iterations in the experiment.
        :param current_iter:
        """
        self.total_train_iters_produced += (current_iter * (self.num_of_gpus * self.batch_size * self.samples_per_iter))

    def get_train_batches(self, total_batches=-1, augment_images=False):
        """
        Returns a training batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length["train"] = total_batches * self.dataset.batch_size
        self.dataset.switch_set(set_name="train", current_iter=self.total_train_iters_produced)
        self.dataset.set_augmentation(augment_images=augment_images)
        self.total_train_iters_produced += (self.num_of_gpus * self.batch_size * self.samples_per_iter)
        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched


    def get_val_batches(self, total_batches=-1, augment_images=False):
        """
        Returns a validation batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['val'] = total_batches * self.dataset.batch_size
        self.dataset.switch_set(set_name="val")
        self.dataset.set_augmentation(augment_images=augment_images)
        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched


    def get_test_batches(self, total_batches=-1, augment_images=False):
        """
        Returns a testing batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['test'] = total_batches * self.dataset.batch_size
        self.dataset.switch_set(set_name='test')
        self.dataset.set_augmentation(augment_images=augment_images)

        if(type(self.dataset)==NLPDataProvider): 
            target_set,target_labels = self.dataset.get_pan_full_target_set()
            for sample_id, sample_batched in enumerate(self.get_dataloader()):
                # validation_tensor = torch.tensor([-1]*self.dataset.batch_size)
                # assert(sample_batched[1] == validation_tensor)
                # assert(sample_batched[3] == validation_tensor)
                target_text_generator = TensorGenerator(target_set,max_length=self.dataset.batch_size)
                target_label_generator = TensorGenerator(target_labels,max_length=self.dataset.batch_size)
                yield (
                    sample_batched[0],
                    target_text_generator, # same samples for all test iterations as required by usecase
                    sample_batched[2],
                    target_label_generator, # same labels for all test iterations as required by usecase
                    sample_batched[-1]
                    )
        else:
            for sample_id, sample_batched in enumerate(self.get_dataloader()):
                yield sample_batched

