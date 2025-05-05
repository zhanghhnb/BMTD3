import numpy as np
import random
from collections import deque
import itertools
import torch

# 修改replaybuffer，使其能够转出历史状态

class ReplayBuffer:

    def __init__(self, size, obs_dim, act_dim):


        self.buffer = deque(maxlen=size)
        # size: 缓冲区的最大容量
        self.max_size = size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_buf = np.zeros((self.max_size, self.obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((self.max_size, self.obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((self.max_size, self.act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((self.max_size, 1), dtype=np.float32)
        self.done_buf = np.zeros((self.max_size, 1), dtype=np.float32)


        self.count = 0

    def sample(self, batch_size, max_hist_len=15):
        # batch = []
        available_samples = self.count - max_hist_len
        batch_size = min(batch_size, available_samples)

        idxs = np.random.choice(np.arange(max_hist_len, self.count), size=batch_size, replace=False)



        HS_BATCH = np.zeros([batch_size, max_hist_len, self.obs_dim])                   #(128, 10, 44)

        HNS_BATCH = np.zeros([batch_size, max_hist_len, self.obs_dim])                  #(128, 10, 44)
        

        HA_BATCH = np.zeros([batch_size, max_hist_len, self.act_dim])                  #(128, 10, 2)

        HNA_BATCH = np.zeros([batch_size, max_hist_len, self.act_dim])                 #(128, 10, 2)

        HSL_BATCH = max_hist_len * np.ones(batch_size)                                    #(128,1)

        HNSL_BATCH = max_hist_len * np.ones(batch_size)                                   #(128,1)


        for i, id in enumerate(idxs):                                             # i: 索引, id: 值

            his_startID = id - max_hist_len


            if len(np.where(self.done_buf[his_startID:id, 0] == 1)[0]) != 0:
                his_startID = his_startID + (np.where(self.done_buf[his_startID:id, 0] == 1)[0][-1]) + 1


            his_seg_len = id - his_startID 

            HSL_BATCH[i] = his_seg_len
            HNSL_BATCH[i] = his_seg_len



            if max_hist_len - his_seg_len == 0:

                HS_BATCH[i] = self.obs_buf[his_startID:id]
                HA_BATCH[i] = self.act_buf[his_startID:id]
                HNS_BATCH[i] = self.obs2_buf[his_startID:id]
                HNA_BATCH[i] = self.act_buf[his_startID+1:id+1]




            else:

                HS_BATCH[i][0:(max_hist_len - his_seg_len)] = self.obs_buf[his_startID]
                HA_BATCH[i][0:(max_hist_len - his_seg_len)] = self.act_buf[his_startID]
                HNS_BATCH[i][0:(max_hist_len - his_seg_len)] = self.obs2_buf[his_startID]
                HNA_BATCH[i][0:(max_hist_len - his_seg_len)] = self.act_buf[his_startID+1]




                HS_BATCH[i][(max_hist_len - his_seg_len)::] = self.obs_buf[his_startID:id]
                HA_BATCH[i][(max_hist_len - his_seg_len)::] = self.act_buf[his_startID:id]
                HNS_BATCH[i][(max_hist_len - his_seg_len)::] = self.obs2_buf[his_startID:id]
                HNA_BATCH[i][(max_hist_len - his_seg_len)::] = self.act_buf[his_startID+1:id+1]




        zhh = {
                    'state':self.obs_buf[idxs],
                    'next_state':self.obs2_buf[idxs],
                    'action':self.act_buf[idxs],
                    'reward':self.rew_buf[idxs],
                    'done':self.done_buf[idxs],
                    'h_state':HS_BATCH,
                    'h_action':HA_BATCH,
                    'h_next_state':HNS_BATCH,
                    'h_next_action':HNA_BATCH,
                    'h_state_length':HSL_BATCH,
                    'h_next_state_length':HNSL_BATCH
                }



        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in zhh.items()}


    def get_length(self):
        return self.count

    def add_sample(self, s, a, r, new_s, done):

        if self.count < self.max_size:
            self.obs_buf[self.count] = s
            self.act_buf[self.count] = a
            self.rew_buf[self.count] = r
            self.obs2_buf[self.count] = new_s
            self.done_buf[self.count] = done
            self.count += 1
        else:
            self.obs_buf[:self.count-1] = self.obs_buf[1:]
            self.act_buf[:self.count-1] = self.act_buf[1:]
            self.rew_buf[:self.count-1] = self.rew_buf[1:]
            self.obs2_buf[:self.count-1] = self.obs2_buf[1:]
            self.done_buf[:self.count-1] = self.done_buf[1:]

            self.obs_buf[self.count-1] = s
            self.act_buf[self.count-1] = a
            self.rew_buf[self.count-1] = r
            self.obs2_buf[self.count-1] = new_s
            self.done_buf[self.count-1] = done

