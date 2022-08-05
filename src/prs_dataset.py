from torch import from_numpy
from torch.utils.data import Dataset
# Dataset class for pytorch
class PRS(Dataset):
    """
    Load PRS scores and phenotypes
    """
    def __init__(self, ids , df_data_x,df_data_y,df_data_p,df_data_a):
        # load data 
        self.ids = from_numpy(ids).int()
        self.prs = from_numpy(df_data_x).float()
        self.pheno = from_numpy(df_data_y).float()
        self.ancestry = from_numpy(df_data_a).int()
        self.pcs = from_numpy(df_data_p).float()

    def __len__(self):
        return len(self.prs)

    def __getitem__(self, index):
        return self.ids[index], self.prs[index], self.pheno[index], self.pcs[index], self.ancestry[index]