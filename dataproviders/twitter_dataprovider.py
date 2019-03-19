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
from dataproviders.encoders import one_hot_encoder

class TwitterDataProvider(Dataset):
    def __init__(self, args):
        """
        A data provider class inheriting from Pytorch's Dataset class. It takes care of creating task sets for
        our few-shot learning model training and evaluation
        :param args: Arguments in the form of a Bunch object. Includes all hyperparameters necessary for the
        data-provider. For transparency and readability reasons to explicitly set as self.object_name all arguments
        required for the data provider, such that the reader knows exactly what is necessary for the data provider/
        """
        self.data_path = args.dataset_path
        self.dataset_name = args.dataset_name
        self.data_loaded_in_memory = False
        self.input_width, self.input_channels = args.input_width, args.input_channels
        self.args = args

        # don't know what this is for
        # a list of numbers
        self.indexes_of_folders_indicating_class = args.indexes_of_folders_indicating_class

        # This is not needed
        # self.reverse_channels = args.reverse_channels
        self.labels_as_int = args.labels_as_int

        # list of train val test split vlaues
        self.train_val_test_split = args.train_val_test_split

        # this makes sense
        self.current_set_name = "train"
        self.num_target_samples = args.num_target_samples
        self.reset_stored_filepaths = args.reset_stored_filepaths # don't know what this is for
        
        val_rng = np.random.RandomState(seed=args.val_seed)
        val_seed = val_rng.randint(1, 999999)
        train_rng = np.random.RandomState(seed=args.train_seed)
        train_seed = train_rng.randint(1, 999999)
        test_rng = np.random.RandomState(seed=args.val_seed)
        test_seed = test_rng.randint(1, 999999)
        args.val_seed = val_seed
        args.train_seed = train_seed
        args.test_seed = test_seed
        
        self.init_seed = {"train": args.train_seed, "val": args.val_seed, 'test': args.val_seed}
        self.seed = {"train": args.train_seed, "val": args.val_seed, 'test': args.val_seed}
        self.num_of_gpus = args.num_of_gpus
        self.batch_size = args.batch_size # task_batch_size

        self.train_index = 0
        self.val_index = 0
        self.test_index = 0

        self.augment_images = False
        self.num_samples_per_class = args.num_samples_per_class
        self.num_classes_per_set = args.num_classes_per_set

        self.rng = np.random.RandomState(seed=self.seed['val'])
        
        # load dataset
        self.datasets = self.load_dataset()

        self.indexes = {"train": 0, "val": 0, 'test': 0}
        self.dataset_size_dict = {
            "train": {key: len(self.datasets['train'][key]) for key in list(self.datasets['train'].keys())},
            "val": {key: len(self.datasets['val'][key]) for key in list(self.datasets['val'].keys())},
            'test': {key: len(self.datasets['test'][key]) for key in list(self.datasets['test'].keys())},
            "target": {key: len(self.datasets['target'][key]) for key in list(self.datasets['target'].keys())}}

        self.label_set = self.get_label_set()
        self.data_length = {name: np.sum([len(self.datasets[name][key])
                                          for key in self.datasets[name]]) for name in self.datasets.keys()}

        print("data", self.data_length)
        self.observed_seed_set = None

    def get_label_from_path(self,filepath):
        return filepath.split('/')[-2]

    def test_file_path(self,filepath):
        if(os.path.isfile(filepath)):
            with open(filepath,'r') as f:
                text = "\n".join(f.readlines())
                if(len(text) > 500):
                    return filepath

        return None

    def save_to_json(self, filename, dict_to_store):
        with open(os.path.abspath(filename), 'w') as f:
            json.dump(dict_to_store, fp=f)

    def load_from_json(self, filename):
        with open(filename, mode="r") as f:
            load_dict = json.load(fp=f)

        return load_dict

    def get_data_paths(self):
        """
        Method that scans the dataset directory and generates class to image-filepath list dictionaries.
        :return: data_image_paths: dict containing class to filepath list pairs.
                 index_to_label_name_dict_file: dict containing numerical indexes mapped to the human understandable
                 string-names of the class
                 label_to_index: dictionary containing human understandable string mapped to numerical indexes
        """
        print("Get text files from", self.data_path)
        data_file_text_path_list_raw = []
        labels = set()
        
        for subdir, dir, files in os.walk(self.data_path):
            for file in files:
                if(file.endswith('.txt')):
                    filepath = os.path.abspath(os.path.join(subdir, file))
                    label = self.get_label_from_path(filepath)
                    if(int(subdir[-3:]) < 40):
                        data_file_text_path_list_raw.append(filepath)
                        labels.add(label)
                
        labels = sorted(labels)
        idx_to_label_name = {idx: label for idx, label in enumerate(labels)}
        label_name_to_idx = {label: idx for idx, label in enumerate(labels)}
        
        data_text_path_dict = {idx: [] for idx in list(idx_to_label_name.keys())}
        target_set_map_dict = {idx: [] for idx in list(idx_to_label_name.keys())}        
        with tqdm.tqdm(total=len(data_file_text_path_list_raw)) as pbar_error:
            with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
                # Process the list of files, but split the work across the process pool to use all CPUs!
                for text_file in executor.map(self.test_file_path,data_file_text_path_list_raw):
                    pbar_error.update(1)
                    if text_file is not None:
                        filename = text_file.split('/')[-1]
                        label = self.get_label_from_path(text_file)
                        idx = label_name_to_idx[label]
                        if('medium' in filename):
                            data_text_path_dict[idx].append(text_file)

                        # use elif to make both dicts disjoint
                        elif('twitter' in filename):
                            target_set_map_dict[idx].append(text_file)

        return data_text_path_dict, idx_to_label_name, label_name_to_idx, target_set_map_dict

    def load_datapaths(self):
        """
        If saved json dictionaries of the data are available, then this method loads the dictionaries such that the
        data is ready to be read. If the json dictionaries do not exist, then this method calls get_data_paths()
        which will build the json dictionary containing the class to filepath samples, and then store them.
        :return: data_image_paths: dict containing class to filepath list pairs.
                 index_to_label_name_dict_file: dict containing numerical indexes mapped to the human understandable
                 string-names of the class
                 label_to_index: dictionary containing human understandable string mapped to numerical indexes
        """
        dataset_dir = os.environ['DATASET_DIR']
        data_path_file = "{}/{}.json".format(dataset_dir, self.dataset_name)
        self.index_to_label_name_dict_file = "{}/map_to_label_name_{}.json".format(dataset_dir, self.dataset_name)
        self.label_name_to_map_dict_file = "{}/label_name_to_map_{}.json".format(dataset_dir, self.dataset_name)
        self.unknown_map_file =  "{}/label_name_to_map_{}.json".format(dataset_dir,self.dataset_name)
        if not os.path.exists(data_path_file):
            self.reset_stored_filepaths = True

        if self.reset_stored_filepaths == True:
            if os.path.exists(data_path_file):
                os.remove(data_path_file)
            self.reset_stored_filepaths = False

        try:
            data_image_paths = self.load_from_json(filename=data_path_file)
            label_to_index = self.load_from_json(filename=self.label_name_to_map_dict_file)
            index_to_label_name_dict_file = self.load_from_json(filename=self.index_to_label_name_dict_file)
            try:
                target_set_map_dict = self.load_from_json(filename=self.unknown_map_file)
            except:
                target_set_map_dict = None
            return data_image_paths, index_to_label_name_dict_file, label_to_index, target_set_map_dict
        except:
            print("Mapped data paths can't be found, remapping paths..")
            data_image_paths, code_to_label_name, label_name_to_code, target_set_map_dict = self.get_data_paths()
            self.save_to_json(dict_to_store=data_image_paths, filename=data_path_file)
            self.save_to_json(dict_to_store=code_to_label_name, filename=self.index_to_label_name_dict_file)
            self.save_to_json(dict_to_store=label_name_to_code, filename=self.label_name_to_map_dict_file)
            if(target_set_map_dict is not None):
                self.save_to_json(dict_to_store=target_set_map_dict, filename=self.unknown_map_file)
            return self.load_datapaths()

    def get_label_set(self):
        """
        Generates a set containing all class numerical indexes
        :return: A set containing all class numerical indexes
        """
        index_to_label_name_dict_file = self.load_from_json(filename=self.index_to_label_name_dict_file)
        return set(list(index_to_label_name_dict_file.keys()))

    def get_index_from_label(self, label):
        """
        Given a class's (human understandable) string, returns the numerical index of that class
        :param label: A string of a human understandable class contained in the dataset
        :return: An int containing the numerical index of the given class-string
        """
        label_to_index = self.load_from_json(filename=self.label_name_to_map_dict_file)
        return label_to_index[label]

    def get_label_from_index(self, index):
        """
        Given an index return the human understandable label mapping to it.
        :param index: A numerical index (int)
        :return: A human understandable label (str)
        """
        index_to_label_name = self.load_from_json(filename=self.index_to_label_name_dict_file)
        return index_to_label_name[index]
    

    def preprocess_data(self, x,method='one_hot_encoder'):
        """
        Preprocesses data such that their shapes match the specified structures
        :param x: A data batch to preprocess
        # a batch will contain an array of array 500 char sentences.
        :return: A preprocessed data batch

        # Perform One Hot encoding
        """
        # try: 
        #     #encoder = encoders[mehthod]
        #     # for i,text_file in enumerate(x): 
        #     x[i] = one_hot_encoder.transform(text_file).toarray()
        #         #x[i] = x[i].reshape(-1,self.input_width,self.input_channels)
        #         # swap features with channels to get a final shape of (#batch,#channels,#features)
        #         #x[i] = np.swapaxes(x[i],1,2) 
            
        #     x = np.vstack(x)
        #     x = x.reshape(-1,self.input_width, self.input_channels)
        #     x = np.swapaxes(x,1,2)
        #     return x

        # except Exception as e:
        #     print(e.args)
        #     return x

        return one_hot_encoder.transform(x.reshape(-1,1)).T.toarray()

    def load_text(self, file_path):
        """
        Given an text filepath and the number of channels to keep, load an image and keep the specified channels
        :param file_path: The text filepath
        :return: an array of 500 char datapoints in the text file.
        """
        if not self.data_loaded_in_memory:
            text_file = open(file_path,'r')
            text = [list(x.replace('\n','')) for x in text_file.readlines() if(len(x) >= 500)]
            text = np.array(text).reshape(-1,500)
            # should I convert this into a one hot encoded vector here itself ? NO!
        else:
            text = file_path

        return text

    def load_batch(self, batch_text_path):
        """
        Load a batch of images, given a list of filepaths
        :param batch_image_paths: A list of filepaths
        :return: A numpy array of images of shape batch, height, width, channels
        """
        text_batch = []

        if self.data_loaded_in_memory:
            for text_path in batch_text_path:
                text_batch.append(text_path)
            text_batch = np.array(text_batch)
            text_batch = self.preprocess_data(text_batch)
            #print(text_batch.shape)
        else:
            text_batch = [self.load_text(file_path=text_path)
                           for text_path in batch_text_path]
            text_batch = np.array(text_batch, dtype=np.float32)
            # text_batch = self.preprocess_data(text_batch)

        return text_batch

    def load_parallel_batch(self, inputs):
        """
        Load a batch of images, given a list of filepaths
        :param batch_file_paths: A list of filepaths
        :return: A numpy array of images of shape batch, height, width, channels
        """
        class_label, batch_file_paths = inputs
        text_batch = []

        # if data was already in memory
        # TODO: what to do if data already in memory ? 
        if self.data_loaded_in_memory:
            for file_path in batch_file_paths:
                text_batch.append(np.copy(file_path))
            text_batch = self.preprocess_data(text_batch)

        else:
            #with tqdm.tqdm(total=1) as load_pbar:
            text_batch = [self.load_text(file_path=file_path)
                           for file_path in batch_file_paths]
                #load_pbar.update(1)

            text_batch = np.vstack(text_batch)

            # text_batch = self.preprocess_data(text_batch)

        return class_label, text_batch

    def load_dataset(self):
        """
        Loads a dataset's dictionary files and splits the data according to the train_val_test_split variable stored
        in the args object.
        :return: Three sets, the training set, validation set and test sets (referred to as the meta-train,
        meta-val and meta-test in the paper)

        load complete dataset in memory
        """
        rng = np.random.RandomState(seed=self.seed['val'])

        data_text_paths, index_to_label_name_dict_file, label_to_index, target_set_map_dict = self.load_datapaths()
        total_label_types = len(data_text_paths)
        num_classes_idx = np.arange(len(data_text_paths.keys()), dtype=np.int32)
        rng.shuffle(num_classes_idx)
        keys = list(data_text_paths.keys())
        values = list(data_text_paths.values())
        new_keys = [keys[idx] for idx in num_classes_idx]
        new_values = [values[idx] for idx in num_classes_idx]
        data_text_paths = dict(zip(new_keys, new_values))
        # data_text_paths = self.shuffle(data_text_paths)
        x_train_id, x_val_id, x_test_id = int(self.train_val_test_split[0] * total_label_types), \
                                            int(np.sum(self.train_val_test_split[:2]) * total_label_types), \
                                            int(total_label_types)
        print(x_train_id, x_val_id, x_test_id)
        x_train_classes = (class_key for class_key in list(data_text_paths.keys())[:x_train_id])
        x_val_classes = (class_key for class_key in list(data_text_paths.keys())[x_train_id:x_val_id])
        x_test_classes = (class_key for class_key in list(data_text_paths.keys())[x_val_id:x_test_id])
        x_train, x_val, x_test = {class_key: data_text_paths[class_key] for class_key in x_train_classes}, \
                                    {class_key: data_text_paths[class_key] for class_key in x_val_classes}, \
                                    {class_key: data_text_paths[class_key] for class_key in x_test_classes},
        dataset_splits = {"train": x_train, "val":x_val , "test": x_test,"target":target_set_map_dict}

        if self.args.load_into_memory is True:

            print("Loading data into RAM")
            x_loaded = {"train": [], "val": [], "test": [], "target":[]}

            for set_key, set_value in dataset_splits.items():
                print("Currently loading into memory the {} set".format(set_key))
                x_loaded[set_key] = {key: np.zeros(len(value), ) for key, value in set_value.items()}
                # for class_key, class_value in set_value.items():
                with tqdm.tqdm(total=len(set_value)) as pbar_memory_load:
                    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
                        # Process the list of files, but split the work across the process pool to use all CPUs!
                        for (class_label, class_images_loaded) in executor.map(self.load_parallel_batch, (set_value.items())):
                            x_loaded[set_key][class_label] = class_images_loaded
                            pbar_memory_load.update(1)

            dataset_splits = x_loaded
            self.data_loaded_in_memory = True

        return dataset_splits

    def shuffle(self, x, rng):
        """
        Shuffles the data batch along it's first axis
        :param x: A data batch
        :return: A shuffled data batch
        """
        indices = np.arange(len(x))
        rng.shuffle(indices)
        x = x[indices]
        return x

    def get_set(self, dataset_name, seed, augment_images=False):
        """
        Generates a task-set to be used for training or evaluation
        :param set_name: The name of the set to use, e.g. "train", "val" etc.
        :return: A task-set containing an image and label support set, and an image and label target set.
        """
        #seed = seed % self.args.total_unique_tasks
        rng = np.random.RandomState(seed)
        selected_classes = rng.choice(list(self.dataset_size_dict[dataset_name].keys()),
                                      size=self.num_classes_per_set, replace=False)
        rng.shuffle(selected_classes)
        
        # inorder to have consistency across test_batches during test phase,
        # we set the episode labels same as the original label of the classes.
        # during the test time we have an assumption that total number of classes
        # will be equal to the num_classes_per_set
        
        episode_labels = range(self.num_classes_per_set)
        if(dataset_name == "test"):
            episode_labels = selected_classes

        class_to_episode_label = {selected_class: episode_label for (selected_class, episode_label) in
                                  zip(selected_classes, episode_labels)}

        x_texts = []
        y_labels = []

        for class_entry in selected_classes:
            choose_support_list = rng.choice(self.dataset_size_dict[dataset_name][class_entry],
                                             size=self.num_samples_per_class, replace=False)
                        
            class_text_samples = []
            class_labels = []
            for sample in choose_support_list:
                choose_samples = self.datasets[dataset_name][class_entry][sample]
                x_class_data = self.load_batch([choose_samples])
               
                class_text_samples.append(torch.tensor(x_class_data,dtype=torch.float32))
                class_labels.append(int(class_to_episode_label[class_entry]))
            
                choose_target_list = rng.choice(self.dataset_size_dict['target'][class_entry],
                                                size=self.num_target_samples, replace=False)
                for sample in choose_target_list:
                    choose_samples = self.datasets["target"][class_entry][sample]
                    x_class_data = self.load_batch([choose_samples])
                
                    class_text_samples.append(torch.tensor(x_class_data,dtype=torch.float32))
                    class_labels.append(int(class_to_episode_label[class_entry]))

            class_text_samples = torch.stack(class_text_samples)
            x_texts.append(class_text_samples)
            y_labels.append(class_labels)

        x_texts = torch.stack(x_texts)
        y_labels = np.array(y_labels, dtype=np.float32)

        support_set_images = x_texts[:, :self.num_samples_per_class]
        support_set_labels = y_labels[:, :self.num_samples_per_class]
        target_set_images = x_texts[:, self.num_samples_per_class:]
        target_set_labels = y_labels[:, self.num_samples_per_class:]
        
        return support_set_images, target_set_images, support_set_labels, target_set_labels, seed

    def __len__(self):
        total_samples = self.data_length[self.current_set_name]
        return total_samples

    def length(self, set_name):
        self.switch_set(set_name=set_name)
        return len(self)

    def set_augmentation(self, augment_images):
        self.augment_images = augment_images

    def switch_set(self, set_name, current_iter=None):
        self.current_set_name = set_name
        if set_name == "train":
            self.update_seed(dataset_name=set_name, seed=self.init_seed[set_name] + current_iter)
        
    def update_seed(self, dataset_name, seed=100):
        self.seed[dataset_name] = seed

    def __getitem__(self, idx):
        support_set_images, target_set_image, support_set_labels, target_set_label, seed = \
            self.get_set(self.current_set_name, seed=self.seed[self.current_set_name] + idx,
                         augment_images=self.augment_images)

        return support_set_images, target_set_image, support_set_labels, target_set_label, seed

    def reset_seed(self):
        self.seed = self.init_seed
