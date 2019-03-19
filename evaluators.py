import torch
import numpy as np
import json
from dataproviders.encoders import one_hot_encoder

class PanEvaluator(object):
    def __init__(self,data_provider,per_model_per_batch_preds,per_model_per_batch_targets):
        self.data_provider = data_provider        
        self.predictions = per_model_per_batch_preds
        self.targets = per_model_per_batch_targets
        self.results = {}

    def get_file_lengths(self,filename):
        with open(filename,'r') as f:
            return sum(1 for x in f.readlines() if len(x) >= 500)


    # def verify_file_and_inputs(self):
    #     #function to verify inputs to the original files
    #     for x in self.idx_to_filename_map
        

    def build_file_map(self):
        dataset = self.data_provider.dataset
        data_paths, idx_to_label, label_to_idx, target_set_map = dataset.load_datapaths()
        self.class_to_file_length_map = {
            k: [self.get_file_lengths(y) for y in v] for k,v in target_set_map.items() 
        }

        self.idx_to_label_map = idx_to_label
        self.idx_to_filename_map = target_set_map
        self.label_to_idx_map = label_to_idx
        self.total_files = sum([len(v) for _,v in target_set_map.items()])

    def rearrange_into_file_predictions(self,predictions,file_ranges):
        file_predictions = [None for x in file_ranges]
        start_idx = 0
        for i,x in enumerate(file_ranges):
            end_idx = start_idx + x
            file_predictions[i] = predictions[start_idx:end_idx]
            start_idx = end_idx

        return file_predictions

    def rearrange_class_predictions(self,file_map):
        for k,v in file_map.items():
            total = sum(v)
            assert(total == sum(self.targets == int(k)))
            class_predictions = self.predictions[np.argwhere(self.targets == int(k))][0]
            file_predictions = self.rearrange_into_file_predictions(class_predictions,v)
            file_map[k] = file_predictions

        return file_map
            

    def get_classification_from_predictions(self,predictions:np.ndarray):
        # on anirudh's recommendation
        # take mean over all prediction probabilities
        # and then return index with max value
        predictions
        return predictions.mean(axis=0).argmax()


    def evaluate(self):
        # assuming only one model
        # assuming only one evaluation_task
        # assuming only 16 tasks
        
        # club all task predictions into one prediction by taking mean of probabilities.
        # this is to enforce an ensembling effect.
        # do it for one model first
        task_accuracies = []
        self.predictions:np.ndarray = np.array(self.predictions[0])
        self.predictions  = self.predictions.mean(axis=0)
        self.targets = self.targets[0]
        self.build_file_map()
        #self.verify_file_and_inputs()
        predictions_file_map = self.rearrange_class_predictions(self.class_to_file_length_map)
        classification_file_map = {
            int(x): [self.get_classification_from_predictions(pred) for pred in v] for x,v in predictions_file_map.items()
        }

        correct_classifications = sum([ sum([1 for p in pred if p == target]) for target,pred in classification_file_map.items()])
        accuracy = correct_classifications/self.total_files
        task_accuracies.append(accuracy)


        
        # result = []
        # for author,file_preds in classification_file_map.items():
        #     for file_idx,pred_idx in enumerate(file_preds):
        #         true_author = self.idx_to_label_map[x]
        #         predicted_author = self.idx_to_label_map[pred_idx]
        #         filename = self.idx_to_filename_map[file_idx].split('/')[-1]
        #         result.append({
        #             "true_author": true_author,
        #             "predicted_author": predicted_author,
        #             "filename": filename
        #         })
        # self.results["classification"] = result
        # self.results["classification_file_map"] = classification_file_map
        # self.results["predictions_to_file_map"] = self.idx_to_filename_map
        # self.results["author_to_idx_map"] = self.label_to_idx_map
        # self.results["predictions_file_map"] = predictions_file_map
        # output_file = "pan18_maml++_pan_5_way_10_shot_problem00002_results.json"
        # with open(output_file,'w') as f:
        #     json.dump(self.results,f)
        print(task_accuracies)
        return task_accuracies
        
            



                
                

        

        
        
        
