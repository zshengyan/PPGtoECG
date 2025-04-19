import torch
import scipy.io
import torch.nn as nn
import time as time
from torch.utils.data import Dataset, DataLoader
from tqdm import *
import matplotlib.pyplot as plt

import models_Sensor as TANN

# Hyper-parameters
INPUT_SIZE = 300 # Cycle length
TEST_NUM = 2

score = torch.zeros(TEST_NUM)
rMSE = torch.zeros(TEST_NUM)
MLoss = nn.MSELoss(reduction='mean')

# Choose device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# PPG2ECG Networks
E = TANN.PPG2ECG(INPUT_SIZE).to(device) # TANN
E = torch.load('Sensor_13.pt', map_location=torch.device('cpu'))
#E = torch.load('Sensor_13.pt').cuda()

# Path of dataset
root = 'E:\\project\\dc_codes\\'


E.eval()
with torch.no_grad():
    for idx in range(TEST_NUM):

        # Path of testing set
        filename = root + 'test\\' + str(idx + 1) + '.mat'

        # Load testing samples
        mat = scipy.io.loadmat(filename)
        ECG_test = mat['seg_E']
        PPG_test = mat['seg_P']

        PPG_test = torch.from_numpy(PPG_test).unsqueeze(1).float().to(device)
        ECG_test = torch.from_numpy(ECG_test).unsqueeze(1).float().to(device)

        # Reconstruct ECG
        E_out = E(PPG_test)
        # Compute correlation coefficient
        #score[idx] = corrcoeff(E_out, ECG_test)

        #print('\n Epoch:', t, 'Avg-rho:', format(torch.mean(score).item(),">8.7f"), 'Med-rho:', format(torch.median(score).item(),">8.7f"), nowt)

        E_out_np = E_out.cpu().detach().numpy() 

        E_out_sample = E_out_np.squeeze()
        plt.plot(E_out_sample)
        plt.title('Reconstructed ECG')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.show()