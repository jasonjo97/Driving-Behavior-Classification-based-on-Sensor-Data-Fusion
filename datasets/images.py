import scipy
import librosa 
import numpy as np

import torch 

class CustomDataset(torch.utils.data.Dataset): 
    """ Convert time series data to image dataset """
    def __init__(self, x, y, class_weights):
        super(CustomDataset, self).__init__()
        
        self.x = x 
        self.y = y
        
        # class weights 
        self.class_weights = class_weights 
        
    def __getitem__(self, idx):        
        # time series normalization 
        x_norm = [((np.array(self.x[i][idx])[~np.isnan(self.x[i][idx])]-
                    np.mean(np.array(self.x[i][idx])[~np.isnan(self.x[i][idx])]))/np.std(np.array(self.x[i][idx])[~np.isnan(self.x[i][idx])])).tolist() 
                      for i in range(len(self.x))]
        
        spectrogram = [librosa.stft(np.array(x_norm[i])[~np.isnan(x_norm[i])], n_fft=128, hop_length=64, win_length=128)
                       for i in range(len(x_norm))]
        magnitude = [np.abs(spectrogram[i]) for i in range(len(spectrogram))]
        log_spectrogram = [librosa.amplitude_to_db(magnitude[i]) for i in range(len(magnitude))]

        # define two features 
        cv = [np.cov(log_spectrogram[i]).tolist() for i in range(len(log_spectrogram))]
        sp = [scipy.signal.periodogram(log_spectrogram[i], nfft=128, scaling='spectrum')[1].tolist() for i in range(len(log_spectrogram))]
        
        x_tensor = torch.from_numpy(np.concatenate((cv,sp), axis=0)).float()
        y_tensor = torch.from_numpy(np.array(self.y[idx])).float()
        weight_tensor = self.class_weights[self.y[idx]] 

        return x_tensor, y_tensor, weight_tensor
    
    def __len__(self):
        return len(self.y)


class StatsRecorder():
    """ Returns mean and standard deviation of image data per channel """
    def __init__(self):
        self.nobservations = 0 # running number of observations  
        
    def update(self, data):
        # initialize statistics and dimensions on first batch 
        if self.nobservations == 0:
            self.mean = data.mean(dim=(0,2,3), keepdims=True)
            self.std = data.std(dim=(0,2,3), keepdims=True)
            self.nobservations = data.shape[0]
        else: 
            # compute mean of new mini-batch  
            new_mean = data.mean(dim=(0,2,3), keepdims=True)
            new_std = data.std(dim=(0,2,3), keepdims=True)
            
            # update number of observations 
            m = self.nobservations * 1.0 
            n = data.shape[0]
            
            # update running statistics 
            self.mean = m/(m+n)*self.mean + n/(m+n)*new_mean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*new_std**2 + \
                        m*n/(m+n)**2 * (self.mean - new_mean)**2
            self.std  = torch.sqrt(self.std) 
            
            # update total number of seen samples 
            self.nobservations += n


class NormalizedDataset(torch.utils.data.Dataset): 
    """ Convert time series data to images with normalization & class/element weights rescaling """
    def __init__(self, x, y, stats, class_weights):
        super(NormalizedDataset, self).__init__()
        
        self.x = x 
        self.y = y
        
        # normalization statistics  
        self.mean = stats[0]
        self.std = stats[1] 
        
        # class weights 
        self.class_weights = class_weights 
        
    def __getitem__(self, idx):
        # time series normalization 
        x_norm = [((np.array(self.x[i][idx])[~np.isnan(self.x[i][idx])]-
                    np.mean(np.array(self.x[i][idx])[~np.isnan(self.x[i][idx])]))/np.std(np.array(self.x[i][idx])[~np.isnan(self.x[i][idx])])).tolist() 
                      for i in range(len(self.x))]
        
        spectrogram = [librosa.stft(np.array(x_norm[i])[~np.isnan(x_norm[i])], n_fft=128, hop_length=64, win_length=128)
                       for i in range(len(x_norm))]
        magnitude = [np.abs(spectrogram[i]) for i in range(len(spectrogram))]
        log_spectrogram = [librosa.amplitude_to_db(magnitude[i]) for i in range(len(magnitude))]

        cv = [np.cov(log_spectrogram[i]).tolist() for i in range(len(log_spectrogram))]
        sp = [scipy.signal.periodogram(log_spectrogram[i], nfft=128, scaling='spectrum')[1].tolist() for i in range(len(log_spectrogram))]
        
        x_tensor = torch.from_numpy(np.concatenate((cv,sp), axis=0)).float()
        y_tensor = torch.from_numpy(np.array(self.y[idx])).float()
        weight_tensor = self.class_weights[self.y[idx]] 
        
        x_tensor = (x_tensor - self.mean.squeeze(0))/self.std.squeeze(0) # apply normalization 
        
        return x_tensor, y_tensor, weight_tensor 
    
    def __len__(self):
        return len(self.y)