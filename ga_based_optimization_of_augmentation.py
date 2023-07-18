#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division

import os
import pygad
import random
import utils
import time
import copy
import pandas as pd
import numpy as np
import torch
import torchvision
from tqdm.notebook import tqdm
from operator import itemgetter
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetimefrom torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms


cudnn.benchmark = True
plt.ion()   # interactive mode

import warnings
warnings.filterwarnings('ignore')


# In[3]:


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# In[4]:


#define the model by extending an ImageClassificationBase class with helper methods for training & validation.
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, device, batch):
        images, labels = batch
#         print(images)
#         print(labels)
        images = images.to(device)
        labels = labels.to(device)
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, device, batch):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
#        return(epoch, result['train_loss'], result['val_loss'], result['val_acc'])
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


# In[5]:


class Net(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.model = torchvision.models.resnet50(num_classes=9)
        state_dict = torch.load('baseline.pth')
        for param in self.model.parameters():
            param.require_grad = False
        for param in self.model.fc.parameters():
            param.require_grad = True
        self.model.load_state_dict(state_dict['model'], strict = False)
    
    def forward(self, xb):
        return (self.model(xb))


# In[6]:


class ImageFolderCustom(ImageFolder):
    def __init__(self, dataset_dir, transforms, chromosome) -> None:
        super().__init__(dataset_dir)
        self.transforms = transforms
        self.chromosome = np.reshape(chromosome, (9, -1)) # CXA shape array (C is the number of classes and A is the number of augmentations)
        
#    def __getitem__(self, index: int) -> Tuple[Any, Any]:
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = transforms.Resize((224,224))(sample)
        augment_probabilities = self.chromosome[target]
        actual_probabilities = [round(random.random(), 1) for _ in range(len(augment_probabilities))]
    
        for act_p, aug_p, tf in zip(actual_probabilities, augment_probabilities, self.transforms):
            if act_p > aug_p:
                #print(act_p, aug_p, tf)
                if(tf=='shear_x'):
                    sample = utils.shear_x(sample)
                elif(tf=='shear_y'):
                    sample = utils.shear_y(sample)
                elif(tf=='translate_x'):
                    sample = utils.translate_x(sample)
                elif(tf=='translate_y'):
                    sample = utils.translate_y(sample)
                elif(tf=='rotate'):
                    sample = utils.rotate(sample)
                elif(tf=='auto_contrast'):
                    sample = utils.auto_contrast(sample)
                elif(tf=='invert'):
                    sample = utils.invert(sample)
                elif(tf=='equalize'):
                    sample = utils.equalize(sample)
                elif(tf=='solarize'):
                    sample = utils.solarize(sample)
                elif(tf=='posterize'):
                    sample = utils.posterize(sample)
                elif(tf=='contrast'):
                    sample = utils.contrast(sample)
                elif(tf=='color'):
                    sample = utils.color(sample)
                elif(tf=='brightness'):
                    sample = utils.brightness(sample)
                elif(tf=='sharpness'):
                    sample = utils.sharpness(sample)
                
                elif(tf=='cutout'):
                    sample = utils.cutout(sample)
        sample = transforms.ToTensor()(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


# In[ ]:


import multiprocessing as mp

try:
    mp.set_start_method('spawn', force=True)
#     print("spawned")
except RuntimeError:
    pass


# In[70]:


@torch.no_grad()
def evaluate(model, val_loader,device):
    model.eval()
    outputs = [model.validation_step(device, batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



def fit_OneCycle(chromosome, chromosome_idx, epochs, max_lr, model, train_loader, val_loader, 
                 test_loader, device, opt_func):
#     fitness_results = {}
    torch.cuda.empty_cache()
    history = []
    optimizer = opt_func(model.parameters(),lr=0.0003,weight_decay=0.0002)
    
    # scheduler for one cycle learniing rate
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))
    #sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer , T_max=10)
    
    best_acc = 0.0
    for epoch in range(1, len(range(epochs+1))):
        # Training Phase 
        model.train()
        train_losses = []
        lrs = []
            
        for batch in train_loader:
            #print(batch)
            loss = model.training_step(device, batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            sched.step()
             
        # Validation phase
        result = evaluate(model, val_loader,device)
        if best_acc< result['val_acc']:
            best_acc = result['val_acc']
            print("best accuracy=", best_acc)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['epoch'] = epoch
        model.epoch_end(epoch, result)
        history.append(result)   
    
    y_pred_list = []
    y_true_list = []
    model.eval()
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            for inp, labels in tepoch:
                inp, labels = inp.to(device), labels.to(device)
                preds = model(inp)
                _, pred_tags = torch.max(preds, dim = 1)   
                correct_pred = (pred_tags == labels).float()
                acc = correct_pred.sum() / len(correct_pred)
                acc = torch.round(acc * 100)
                tepoch.set_postfix(accuracy = acc.item())
                y_pred_list.append(pred_tags.cpu().numpy())
                y_true_list.append(labels.cpu().numpy())

    # flatten prediction and true lists
    flat_pred = []
    flat_true = []
    for i in range(len(y_pred_list)):
        for j in range(len(y_pred_list[i])):
            flat_pred.append(y_pred_list[i][j])
            flat_true.append(y_true_list[i][j])
    matrix = confusion_matrix(flat_true, flat_pred)
    mean_per_class = (matrix.diagonal()/matrix.sum(axis=1)).mean()
    per_class = np.round(matrix.diagonal()/matrix.sum(axis=1), 4)
    
    
    with open('perclass_accuracies.csv', 'a') as fd:
        fd.write(f'{per_class} , {mean_per_class}\n')
    print("Mean_per_class", mean_per_class)
    
    plt.figure()
    sns.heatmap(matrix, annot=True, fmt='').set(title="confusion matrix", xlabel="Predicted Label", ylabel="True Label")
    plt.savefig('matrix_%.6f.png' %mean_per_class)
    plt.close()
    
    torch.save(model.state_dict(), ('model_%.16f.pth' %mean_per_class))
    torch.cuda.empty_cache()
    return history, mean_per_class



# init_pop  = [generate_pop(135) for _ in range(99)]
# init_pop.append([0.0]*135)

# In[72]:


def fitness_func(chromosome, chromosome_idx ):
    '''
    chromosome -- NxCxA (N is the population, C is the number of classes, A is the number possible augmentations)
    '''
#    torch.multiprocessing.set_start_method('spawn')
#     try:
#         torch.multiprocessing.set_start_method('spawn')
#     except RuntimeError:
#         pass
#     process_id = (chromosome_idx%8)
    current_device = torch.device(('cuda:%d')%(chromosome_idx%4))
    train_dataset_dir = 'soybean_testandtrain/train'
    test_dataset_dir = 'soybean_testandtrain/val'
    
    transforms1 = ['auto_contrast','invert', 'equalize', 'solarize', 'posterize', 'contrast', 'color',
                   'brightness','sharpness', 'shear_x', 'shear_y', 'translate_x', 'translate_y', 'rotate',
                   'cutout']
    test_transforms=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = ImageFolderCustom(train_dataset_dir, transforms=transforms1, chromosome=chromosome)
    test_set = ImageFolder(test_dataset_dir, transform=test_transforms)
    lengths = [int(np.ceil(0.85*len(dataset))),
           int(np.floor(0.15*len(dataset)))]
    train_set, valid_set = data.random_split(dataset, lengths)
    batch_size = 256    
    train_dl = DataLoader(train_set, batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valid_dl = DataLoader(valid_set, batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_dl = DataLoader(test_set, batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    epochs = 5
    max_lr = 0.001
    model = to_device(Net(), device=current_device)
    history, mpca = fit_OneCycle(chromosome, chromosome_idx, epochs, max_lr, model, train_dl, valid_dl, test_dl, 
                                 current_device, torch.optim.AdamW)
    return mpca



num_generations = 3
num_parents_mating = 40
# initial_pop = init_pop

sol_per_pop = 100
num_genes = 135
keep_elitism = 30

init_range_low = 0.0
init_range_high = 1.0

parent_selection_type = "sss"
#keep_parents = 5

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 5


# best_results = []
last_fitness = 0
def on_generation(loaded_ga_instance):
    global last_fitness
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Population = {population}".format(population = ga_instance.population))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
    print("Change     = {change}".format(change=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness))
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
    

    
def main():
#     loaded_ga_instance = pygad.load(filename='ga_instance')
    
    ga_instance = pygad.GA(sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
#                        initial_population = init_pop,
                       gene_type=[float, 1],
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_elitism = keep_elitism,
#                       keep_parents=keep_parents,
                       crossover_type=crossover_type,  
                       mutation_type=mutation_type,
                       mutation_by_replacement=True,
#                       gene_space = [{'low': 0, 'high': 1.0}],
                       mutation_percent_genes=mutation_percent_genes,
                       random_mutation_min_val=0.0,
                       random_mutation_max_val=1.0,
                       on_generation=on_generation,
                       save_best_solutions=True,
                       save_solutions=True,
                       parallel_processing=["process", 4])
    t1 = time.time()
    ga_instance.run()
    t2 = time.time()
    print("Time is", t2-t1)
    pd.DataFrame(ga_instance.best_solutions).to_csv('best_solutions.csv')
    
    pd.DataFrame(ga_instance.solutions).to_csv('solutions.csv')
    ga_instance.plot_fitness(save_dir='fitness.png')
   
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    print("Solution", solution)
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    ga_instance.save(filename='ga_instance')
    pd.DataFrame(ga_instance.best_solutions_fitness).to_csv('best_solutions_fitness.csv')
    

if __name__ == '__main__':
    main()





