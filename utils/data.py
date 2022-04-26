import os 
import numpy as np 


def read_x_data(filename): 
    with open(filename, 'r') as f: 
        x_data = [list(map(float,line.split()[1:])) for line in f.read().splitlines()]
    return x_data

def read_y_data(filename): 
    with open(filename, 'r') as f: 
        drivers, y_labels = [], []
        for line in f.read().splitlines():     
            drivers.append(line.split()[0])
            y_labels.append(int(line.split()[-1]))
        return drivers, y_labels 

def load_data(folder): 
    x_folder = folder + 'X_values_DL'
    y_folder = folder + 'Y_values_DL/y_label.txt'
    
    X = []
    
    for file in os.listdir(x_folder):
        if file.endswith('.txt'): 
            X.append(read_x_data(x_folder + '/' + file))
        
    drivers, y = read_y_data(y_folder) 
    
    return X, y, drivers

def load_data_vision(folder): 
    x_folder = folder + 'X_values_DL/Vision'
    
    X = []
    
    for file in os.listdir(x_folder):
        X.append(read_x_data(x_folder + '/' + file))
    
    return X

def nan_padding(lst, max_length):
    for i in range(len(lst)): 
        if max_length > len(lst[i]): lst[i].extend([np.nan]*(max_length-len(lst[i]))) 
        else: lst[i] = lst[i][:max_length]
        
    return lst

def find_maxlength(x):
    max_length = 0 

    for i in range(len(x)): 
        if len(x[i]) > 25000: 
            continue   
        if max_length < len(x[i]): 
            max_length = len(x[i])

    return max_length