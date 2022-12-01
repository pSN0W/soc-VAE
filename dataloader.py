import torch
import numpy as np
from torch.utils import data as Data
import random as rnd

class SocialVAEDataset(Data.Dataset):
    def __init__(self,
                file_name:str,
                ob_horizon:int=8,
                pred_horizon:int=12,
                flip: bool=False, batch_first: bool=False, scale: bool=False
                ) -> None:
        super().__init__()
        self.file_name = file_name
        self.ob_horizon = ob_horizon
        self.pred_horizon = pred_horizon
        self.horizon = self.ob_horizon + self.pred_horizon
        self.flip = flip
        self.batch_first = batch_first
        self.scale = scale
        self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        
        self.data = np.array(self.get_data_from_file(),dtype=object)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
        
    def collate_fn(self,batch):
        X, Y, NEIGHBOR = [],[],[]
        for item in batch:
            hist, future, neighbor = item[0], item[1], item[2]

            hist_shape = hist.shape
            neighbor_shape = neighbor.shape
            hist = np.reshape(hist, (-1, 2))
            neighbor = np.reshape(neighbor, (-1, 2))
            
            # Data Augmentation
            if self.flip:
                if rnd.randint(0,2):
                    hist[..., 1] *= -1
                    future[..., 1] *= -1
                    neighbor[..., 1] *= -1
                if rnd.randint(0,2):
                    hist[..., 0] *= -1
                    future[..., 0] *= -1
                    neighbor[..., 0] *= -1
            
            if self.scale:
                s = rnd.normalvariate(0,1)*0.05 + 1 # N(1, 0.05)
                hist = s * hist
                future = s * future
                neighbor = s * neighbor
            
            hist = np.reshape(hist, hist_shape)
            neighbor = np.reshape(neighbor, neighbor_shape)

            X.append(hist)
            Y.append(future)
            NEIGHBOR.append(neighbor)
        
        n_neighbors = [n.shape[1] for n in NEIGHBOR]
        max_neighbors = max(n_neighbors) 
        if max_neighbors != min(n_neighbors):
            NEIGHBOR = [
                np.pad(neighbor, ((0, 0), (0, max_neighbors-n), (0, 0)), 
                "constant", constant_values=1e9)
                for neighbor, n in zip(NEIGHBOR, n_neighbors)
            ]
        
        stack_dim = 0 if self.batch_first else 1
        x = np.stack(X, stack_dim)
        y = np.stack(Y, stack_dim)
        neighbor = np.stack(NEIGHBOR, stack_dim)

        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        neighbor = torch.tensor(neighbor, dtype=torch.float32, device=self.device)
        return x, y, neighbor
        
        
    def get_data_from_file(self):
        with open(self.file_name,'r') as f:
            data = self.process_file(f)
        data = self.add_diff_feature(data)
        
        # fin data with neighbours
        fin_data = []
        hor = self.horizon - 1
        time = np.sort(list(data.keys()))
        e = len(time)
        for tid0 in range(e-hor):
            tid1 = tid0 + hor
            t0 = time[tid0]
            idx = list(data[t0].keys())
            idx_all = list(data[t0].keys())
            for tid in range(tid0+1,tid1+1):
                t = time[tid]
                idx_cur = list(data[t].keys())
                idx = np.intersect1d(idx,idx_cur)
                if len(idx) == 0: break
                idx_all.extend(idx_cur)
                
            if len(idx):
                data_dim = 6
                neighbor_idx = np.setdiff1d(idx_all, idx)
                if len(idx) == 1 and len(neighbor_idx) == 0:
                    agents = np.array([
                        [data[time[tid]][idx[0]][:data_dim]] + [[1e9]*data_dim]
                        for tid in range(tid0, tid1+1, 1)
                    ]) # L x 2 x 6
                else:
                    agents = np.array([
                        [data[time[tid]][i][:data_dim] for i in idx] +
                        [data[time[tid]][j][:data_dim] if j in data[time[tid]] else [1e9]*data_dim for j in neighbor_idx]
                        for tid in range(tid0, tid1+1)
                    ])  # L X N x 6
                    
            for i in range(len(idx)):
                hist = agents[:self.ob_horizon,i]  # L_ob x 6
                future = agents[self.ob_horizon:self.horizon,i,:2]  # L_pred x 2
                neighbor = agents[:self.horizon, [d for d in range(agents.shape[1]) if d != i]] # L x (N-1) x 6
                fin_data.append((hist, future, neighbor))
                
        items = []
        for hist, future, neighbor in fin_data:
            hist = np.float32(hist)
            future = np.float32(future)
            neighbor = np.float32(neighbor)
            items.append((hist, future, neighbor))
        return items
    
    def add_diff_feature(self,data):
        # check if the time interval in observation is same
        time = np.sort(list(data.keys()))
        time_diff = time[1:] - time[:-1]
        assert np.all(time_diff==time_diff[0]), 'Add support for different time interval'
        
        # remove all the agents appearing in only one frame
        for tid,t in enumerate(time):
            remove_agents = []
            for idx in data[t].keys():
                t0 = time[tid-1] if tid != 0 else None
                t1 = time[tid+1] if tid+1 != len(time) else None
                if (t0 is None or t0 not in data or idx not in data[t0]) and \
                (t1 is None or t1 not in data or idx not in data[t1]):
                    remove_agents.append(idx)
            
            for idx in remove_agents:
                data[t].pop(idx)
                
        # adding feature velocity
        for t in range(len(time)-1):
            t0 = time[t]
            t1 = time[t+1]
            for i in data[t1]:
                if i not in data[t0]:
                    data[t1][i].insert(2, 0)
                    data[t1][i].insert(3, 0)
                    continue
                x0 = data[t0][i][0]
                y0 = data[t0][i][1]
                x1 = data[t1][i][0]
                y1 = data[t1][i][1]
                vx, vy = x1-x0, y1-y0
                data[t1][i].insert(2, vx)
                data[t1][i].insert(3, vy)
                # using the next one for the one we can't calculate
                if tid < 0 or i not in data[time[tid-1]]:
                    data[t0][i].insert(2, vx)
                    data[t0][i].insert(3, vy)
        # adding feature acceleration
        for t in range(len(time)-1):
            t0 = time[t]
            t1 = time[t+1]
            for i in data[t1]:
                if i not in data[t0]:
                    data[t1][i].insert(4, 0)
                    data[t1][i].insert(5, 0)
                    continue
                vx0 = data[t0][i][2]
                vy0 = data[t0][i][3]
                vx1 = data[t1][i][2]
                vy1 = data[t1][i][3]
                ax, ay = vx1-vx0, vy1-vy0
                data[t1][i].insert(4, ax)
                data[t1][i].insert(5, ay)
                # using the next one for the one we can't calculate
                if tid < 0 or i not in data[time[tid-1]]:
                    data[t0][i].insert(4, ax)
                    data[t0][i].insert(5, ay)
                    
        return data
        
    def process_file(self,f):
        data = {}
        for row in f.readlines():
            item = row.split()
            if not item: continue
            t,ped,x,y = int(float(item[0])),int(float(item[1])),float(item[2]),float(item[3])
            data.setdefault(t,{})
            data[t][ped] = [x, y, None]
        return data
    
dataset1 = SocialVAEDataset('./train/biwi_hotel.txt')
dataloader1 = Data.DataLoader(dataset=dataset1,batch_size=2,shuffle=False,collate_fn=dataset1.collate_fn)