import torch
from torch.utils import data
import crispr_attn
import attention_setting
import os

class MyDataset(data.Dataset):

  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        drug_combine_file = os.path.join(attention_setting.data_folder, str(ID) + '.pt')
        # Load data and get label
        try:
            X = torch.load(drug_combine_file)
        except:
            crispr_attn.logger.error("Fail to get {}".format(ID))
            raise
        y = self.labels[ID]

        return X, y
