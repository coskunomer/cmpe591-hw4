import torch
from torch.utils.data import Dataset
import pickle
import random
import numpy as np

class CNMPDataset(Dataset):
    def __init__(self, path="cnmp_dataset.pkl"):
        with open(path, "rb") as f:
            self.trajectories = pickle.load(f)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx] 

        T = traj.shape[0]
        h = traj[0, 4] 

        idxs = list(range(T))
        random.shuffle(idxs)
        n_context = random.randint(1, T - 1)
        n_target = random.randint(1, T - n_context)

        context_idxs = idxs[:n_context]
        target_idxs = idxs[n_context:n_context + n_target]

        def extract(indices):
            x = []
            y = []
            for i in indices:
                t = i / (T - 1)  # normalize time
                e_y, e_z, o_y, o_z, _ = traj[i]
                x.append([t, h])
                y.append([e_y, e_z, o_y, o_z])
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

        context_x, context_y = extract(context_idxs)
        target_x, target_y = extract(target_idxs)

        observation = torch.cat([context_x, context_y], dim=-1)  # shape: (n_context, 6)

        return observation.unsqueeze(0), target_x.unsqueeze(0), target_y.unsqueeze(0)
