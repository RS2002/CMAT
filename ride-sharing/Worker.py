import numpy as np
import torch
from model import CMAT
from joblib import Parallel, delayed
import tqdm
import pickle
from torch.cuda.amp import GradScaler
from scipy.spatial import KDTree
import torch.nn as nn


# torch.autograd.set_detect_anomaly(True)

INF = 1e8


class ValueNorm(nn.Module):
    """ Normalize a vector of observations - across the first norm_axes dimensions"""

    def __init__(self, input_shape, norm_axes=1, beta=0.99999, per_element_update=False, epsilon=1e-5,
                 device=torch.device("cpu")):
        super(ValueNorm, self).__init__()

        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))

        if self.per_element_update:
            batch_size = np.prod(input_vector.size()[:self.norm_axes])
            weight = self.beta ** batch_size
        else:
            weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector):
        # Make sure input is float32
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.running_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
        out = out.squeeze(0)
        
        return out

    def denormalize(self, input_vector):
        """ Transform normalized data back into original distribution """
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
        out = out.squeeze(0)
        # out = out.cpu().numpy()

        return out

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)


def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    """ Generalized Advantage Estimation """
    advantages = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value[t] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    return advantages


class Buffer():
    def __init__(self,capacity = 1e5, episode_capacity = 10):
        super().__init__()
        self.reset()

    def reset(self, capacity = None, episode_capacity = None):
        self.num = 0

        self.worker_state = []
        self.order_state = []
        self.order_num = []
        self.pooling_order = []
        self.action = []
        self.p_log = []

        self.worker_state_next = []
        self.order_state_next = []
        self.order_num_next = []
        self.pooling_order_next = []
        self.action_next = []

        self.reward = []

        self.img = []
        self.img_next = []
        self.done = []

    def append(self, experience, episode=0):
        self.num += 1

        state, action, reward, state_next, action_next = experience
        action, p_log = action

        self.worker_state.append(torch.from_numpy(state[0]))
        self.order_state.append(torch.from_numpy(state[1]))
        self.order_num.append(torch.from_numpy(state[2]))
        self.pooling_order.append(torch.from_numpy(state[3]))
        self.img.append(torch.from_numpy(state[4]))
        self.action.append(torch.tensor(action))
        self.p_log.append(p_log)
        if action_next is not None:
            action_next, _ = action_next
            self.worker_state_next.append(torch.from_numpy(state_next[0]))
            self.order_state_next.append(torch.from_numpy(state_next[1]))
            self.order_num_next.append(torch.from_numpy(state_next[2]))
            self.pooling_order_next.append(torch.from_numpy(state_next[3]))
            self.img_next.append(torch.from_numpy(state_next[4]))
            self.action_next.append(torch.tensor(action_next))
            self.done.append(True)
        else:
            self.worker_state_next.append(state_next[0])
            self.order_state_next.append(state_next[1])
            self.order_num_next.append(state_next[2])
            self.pooling_order_next.append(state_next[3])
            self.img_next.append(state_next[4])
            self.action_next.append(action_next)
            self.done.append(False)

        # self.reward.append(reward)
        self.reward.append(reward/1000)



    def sample(self):
        indices = np.arange(self.num)

        worker_state = [self.worker_state[i] for i in indices]
        order_state = [self.order_state[i] for i in indices]
        order_num = [self.order_num[i] for i in indices]
        pool_order = [self.pooling_order[i] for i in indices]
        action = [self.action[i] for i in indices]
        worker_state_next = [self.worker_state_next[i] for i in indices]
        order_state_next = [self.order_state_next[i] for i in indices]
        order_num_next = [self.order_num_next[i] for i in indices]
        pool_order_next = [self.pooling_order_next[i] for i in indices]
        action_next = [self.action_next[i] for i in indices]
        reward = [self.reward[i] for i in indices]

        img = [self.img[i] for i in indices]
        img_next = [self.img_next[i] for i in indices]

        p_log = [self.p_log[i] for i in indices]
        done = [self.done[i] for i in indices]

        return img, worker_state, order_state, order_num, pool_order, action, p_log, reward, img_next, worker_state_next, order_state_next, order_num_next, pool_order_next, action_next, done

def calculate_entropy(log_probs, probs):
    entropy = -torch.sum(probs * log_probs, dim=1)
    return torch.mean(entropy)

def norm(order_state, worker_state, history_order_state, lat_min = 40.68878421555262, lat_max = 40.875967791801536, lon_min = -74.04528828347375, lon_max = -73.91037864632285, simulation_time = 60, max_capacity = 3):
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    if isinstance(order_state, torch.Tensor):
        worker_state, history_order_state, order_state = worker_state.clone(), history_order_state.clone(), order_state.clone()
    else:
        worker_state, history_order_state, order_state = worker_state.copy(), history_order_state.copy(), order_state.copy()

    # 1. lat & lon
    order_state[:,0] = (order_state[:,0] - lat_min) / lat_range
    order_state[:,2] = (order_state[:,2] - lat_min) / lat_range
    order_state[:,1] = (order_state[:,1] - lon_min) / lon_range
    order_state[:,3] = (order_state[:,3] - lon_min) / lon_range

    worker_state[:,0] = (worker_state[:,0] - lat_min) / lat_range
    worker_state[:,1] = (worker_state[:,1] - lon_min) / lon_range


    history_order_state[:,:,0] = (history_order_state[:,:,0] - lat_min) / lat_range * (history_order_state[:,:,0] != 0)
    history_order_state[:,:,1] = (history_order_state[:,:,1] - lon_range) / lon_range * (history_order_state[:,:,1] != 0)

    # 2. time
    worker_state[:, 3] = worker_state[:, 3] / simulation_time
    worker_state[:, 5] = worker_state[:, 5] / simulation_time
    worker_state[:, 6] = worker_state[:, 6] / simulation_time

    order_state[:,4] = order_state[:,4] / simulation_time

    history_order_state[:,:,2] = history_order_state[:,:,2] / simulation_time
    history_order_state[:,:,3] = history_order_state[:,:,3] / simulation_time
    history_order_state[:,:,4] = history_order_state[:,:,4] / simulation_time

    # 3. capacity
    worker_state[:, 2] = worker_state[:, 2] / max_capacity

    return order_state, worker_state, history_order_state

def img_norm(img):
    if isinstance(img, torch.Tensor):
        img = img.clone()
    else:
        img = img.copy()

    img[:,2:] = img[:,2:] / 10

    return img


lat_min = 40.68878421555262
lat_max = 40.875967791801536
lon_min = -74.04528828347375
lon_max = -73.91037864632285

class Worker():
    def __init__(self, buffer, lr=0.0001, gamma=0.99, max_step=60, num=1000, device=None, zone_table_path = "./data/Manhattan_dic.pkl", model_path = None, njobs = 24, bi_direction = True, dropout = 0.0, compression = False, pretrain_model_path = None, rand_init = False, restrict = False, noise_type = 0, actor_rate = 0.1, finetune = None):
        super().__init__()
        self.buffer = buffer

        self.gamma = gamma
        self.device = device
        self.max_step = max_step
        self.num = num
        self.rand_init = rand_init
        self.restrict = restrict
        self.noise_type = noise_type
        self.actor_rate = actor_rate
        with open(zone_table_path, 'rb') as f:
            self.zone_dic = pickle.load(f)
        # self.zone_lookup = self.zone_dic["zone_num"]
        self.coordinate_lookup_lat = np.array(self.zone_dic["centroid_lat"])
        self.coordinate_lookup_lon = np.array(self.zone_dic["centroid_lon"])
        self.zone_map = np.array(self.zone_dic["map"])

        self.AC_training = CMAT(state_size=7, enroute_order_size=5, new_order_size=5, grid_size=11, grid_num=len(self.coordinate_lookup_lat), hidden_dim=64, consensus_len=5, finetune=finetune).to(self.device)

        # self.AC_target = CMAT(state_size=7, enroute_order_size=5, new_order_size=5, grid_size=11, grid_num=len(self.coordinate_lookup_lat), hidden_dim=64, consensus_len=5).to(self.device)
        #
        # self.update_target(tau=1.0)
        # for param in self.AC_target.parameters():
        #     param.requires_grad = False
        # self.AC_target.eval()


        self.lat_min = 40.68878421555262
        self.lat_max = 40.875967791801536
        self.lon_min = -74.04528828347375
        self.lon_max = -73.91037864632285
        self.lon_range, self.lat_range = self.lon_max-self.lon_min, self.lat_max-self.lat_min

        # KL 自适应相关
        self.target_kl = 0.01
        self.beta = 1.0
        self.beta_min = 1/16
        self.beta_max = 16.0
        self.beta_adjust_factor = 1.5
        self.early_stop_kl = 0.03


        self.load(model_path,self.device)
        print('Platform total parameters:', sum(p.numel() for p in self.AC_training.parameters() if p.requires_grad))

        self.optim = torch.optim.Adam(self.AC_training.parameters(), lr=lr, weight_decay=0)
        self.schedule = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=0.99)

        self.njobs = njobs
        self.scaler = GradScaler()
        self.reset()

        self.value_normalizer = ValueNorm(1, device=self.device)

    # def update_target(self, tau=1.0):
    #     for target_param, train_param in zip(self.AC_target.parameters(), self.AC_training.parameters()):
    #         target_param.data.copy_(tau * train_param.data + (1.0 - tau) * target_param.data)


    def save(self, path):
        torch.save(self.AC_training.state_dict(), path)


    def load(self, path=None, device=torch.device("cpu")):
        if path is not None:
            self.AC_training.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            # self.AC_target.load_state_dict(torch.load(path, map_location=device, weights_only=True))


    def reset(self, capacity = 3, train=True):
        if train:
            self.AC_training.train()
        else:
            self.AC_training.eval()
        torch.set_grad_enabled(False)

        self.is_train = train

        '''
        observation space
        0,1: current lat,lon (required to be normalized before inputting to the network, following lat and lon remain same)
        2: remaining order place
        3: remaining picking time
        4: state -- 0 allows to pick up new orders, 1 does not (because picking up the order that doesn't allow pooling or the capacity is full)
        5: current time
        6: idle time
        '''
        self.observe_space = np.zeros([self.num, 7])
        self.observe_space[:,2] = capacity

        '''
        current orders
        0,1: drop-off lat,lon
        2: remaining transportation time (approximated)
        3: total transportation time (approximated)
        4: detour time (current)
        '''
        self.current_orders = np.zeros([self.num, capacity, 5])
        self.current_order_num = np.zeros([self.num])

        if self.rand_init:
            # allocate a initial location randomly from valid zone
            self.observe_space[:, 0] = np.random.random([self.num]) * (lat_max - lat_min) + lat_min
            self.observe_space[:, 1] = np.random.random([self.num]) * (lon_max - lon_min) + lon_min
        else:
            # allocate a initial location randomly from valid zone
            random_integers = np.random.randint(0, len(self.coordinate_lookup_lat), size=(self.num))
            self.observe_space[:, 0] = self.coordinate_lookup_lat[random_integers]
            self.observe_space[:, 1] = self.coordinate_lookup_lon[random_integers]

        # some records for simulation
        self.travel_route = [[] for _ in range(self.num)]
        self.travel_time = [[] for _ in range(self.num)]
        self.experience = []
        self.Pass_Travel_Time = []
        self.Detour_Time = []

        self.img = np.zeros([len(self.coordinate_lookup_lat),11])
        # self.img[:, 0] = self.coordinate_lookup_lat
        # self.img[:, 1] = self.coordinate_lookup_lon
        # self.img[:, 0] = (self.img[:,0] - self.lat_min) / self.lat_range
        # self.img[:, 1] = (self.img[:,1] - self.lon_min) / self.lon_range
        self.tree = KDTree(self.img[:, :2])


    def observe(self, order, current_time, exploration_rate=0):
        pid = order['PULocationID']
        did = order['DOLocationID']
        pid = self.zone_map[pid - 1]
        did = self.zone_map[did - 1]
        minute = order['minute']
        plat, plon = self.coordinate_lookup_lat[pid], self.coordinate_lookup_lon[pid]
        dlat, dlon = self.coordinate_lookup_lat[did], self.coordinate_lookup_lon[did]
        minute = np.array(minute).reshape(-1, 1)
        plat = np.array(plat).reshape(-1, 1)
        plon = np.array(plon).reshape(-1, 1)
        dlat = np.array(dlat).reshape(-1, 1)
        dlon = np.array(dlon).reshape(-1, 1)
        order = np.concatenate([plat, plon, dlat, dlon, minute], axis=-1)

        self.observe_space[:,5] = current_time

        # # 0. construct image
        # order_lat = order[:,0]
        # order_lon = order[:,1]
        # worker_lat = self.observe_space[:,0]
        # worker_lon = self.observe_space[:,1]
        # order_lat = (order_lat - self.lat_min) / self.lat_range
        # order_lon = (order_lon - self.lon_min) / self.lon_range
        # worker_lat = (worker_lat - self.lat_min) / self.lat_range
        # worker_lon = (worker_lon - self.lon_min) / self.lon_range
        # order_cor = np.column_stack((order_lat, order_lon))
        # worker_cor = np.column_stack((worker_lat, worker_lon))
        # _, order_pos = self.tree.query(order_cor)
        # _, worker_pos = self.tree.query(worker_cor)
        # worker_available = self.observe_space[:,4]
        # worker_seat = self.observe_space[:,2]
        # self.img[:,5:] = self.img[:,2:-3]
        # self.img[:,2:5] = 0
        # np.add.at(self.img[:,2], (order_pos.astype(int)), 1)
        # np.add.at(self.img[:,3], (worker_pos[worker_available==0].astype(int)), 1)
        # np.add.at(self.img[:,4], (worker_pos.astype(int)), worker_seat)
        img = torch.tensor(self.img).to(self.device)
        # img = img_norm(img)


        # 1. calculate q-value
        x1, x2, x3 = norm(order, self.observe_space, self.current_orders)
        x1, x2, x3 = torch.tensor(x1).to(self.device), torch.tensor(x2).to(self.device), torch.tensor(x3).to(self.device)

        if self.noise_type != 1:
            _, _, p_matrix = self.AC_training(img, x1, x2, x3, torch.from_numpy(self.current_order_num).to(self.device))

            p_log = p_matrix.cpu().detach().numpy().copy()

            # 2. epsilon-greedy explore
            if self.noise_type == 0:  # BSC
                exploration_matrix = torch.rand_like(p_matrix)
                p_matrix[exploration_matrix < exploration_rate] = INF
            elif self.noise_type == 1:  # Gaussian-1
                q_mean = torch.mean(p_matrix[self.observe_space[:, 4] == 0])
                scale = 2 * q_mean * exploration_rate
                noise = torch.randn_like(p_matrix) * scale
                p_matrix = p_matrix + noise
            elif self.noise_type == 2:  # Gaussian-2
                gaussian_noise = torch.randn_like(p_matrix) * exploration_rate
                p_matrix = p_matrix + gaussian_noise
            else:  # Uniform
                uni_noise = torch.rand_like(p_matrix) * exploration_rate * 3.5
                p_matrix = p_matrix + uni_noise
        else:
            if exploration_rate == 0:
                _, _, p_matrix = self.AC_training(img, x1, x2, x3, torch.from_numpy(self.current_order_num).to(self.device))
                p_log = p_matrix.cpu().detach().numpy().copy()
            else:
                _, _, p_log, p_matrix = self.AC_training(img, x1, x2, x3, torch.from_numpy(self.current_order_num).to(self.device),exploration_rate=exploration_rate)
                p_log = p_log.cpu().detach().numpy().copy()

        p_matrix[self.observe_space[:, 4] == 1] = -INF

        # 3. add distance restriction
        if self.restrict:
            worker_pos = self.observe_space[:, :2]
            order_pos = order[:, :2]
            dis = np.sqrt(np.sum((worker_pos[:, np.newaxis, :] - order_pos[np.newaxis, :, :]) ** 2, axis=-1))
            p_matrix[dis > 0.03] -= INF / 2

        return p_matrix.cpu().detach().numpy(), order, p_log


    def train(self, batch_size=8, train_times=15, show_pbar=True):

        eps_clip = 0.2
        loss_list =[]


        torch.set_grad_enabled(True)
        if show_pbar:
            pbar = tqdm.tqdm(range(train_times))
        else:
            pbar = range(train_times)
        img, worker_state, order_state, order_num, pool_order, action, p_log_old, reward, img_next, worker_state_next, order_state_next, order_num_next, pool_order_next, action_next, done = self.buffer.sample()

        p_log_old, reward = torch.tensor(p_log_old).to(self.device), torch.tensor(reward).to(self.device)
        v_value = torch.zeros([len(worker_state)]).to(self.device)
        v_next = torch.zeros([len(worker_state)]).to(self.device)

        for i in range(len(worker_state)):
            img_temp, worker_state_temp, order_state_temp, order_num_temp, pool_order_temp, action_temp, p_log_old_temp, reward_temp, img_next_temp, worker_state_next_temp, order_state_next_temp, order_num_next_temp, pool_order_next_temp, action_next_temp = img[i], worker_state[i], order_state[i], order_num[i], pool_order[i], action[i], p_log_old[i], reward[i], img_next[i], worker_state_next[i], order_state_next[i], order_num_next[i], pool_order_next[i], action_next[i]
            img_temp, worker_state_temp, order_state_temp, order_num_temp, pool_order_temp, action_temp = img_temp.to(
                self.device), worker_state_temp.to(self.device), order_state_temp.to(self.device), order_num_temp.to(
                self.device), pool_order_temp.to(self.device), action_temp.to(self.device)
            x1, x2, x3 = norm(pool_order_temp, worker_state_temp, order_state_temp)
            img_temp = img_norm(img_temp)
            v, _, _ = self.AC_training(img_temp, x1, x2, x3, order_num_temp)
            v = v.detach()
            v_value[i] = v
            if i != 0:
                v_next[i-1] = v

        advantage = compute_gae(reward, self.value_normalizer.denormalize(v_value),  self.value_normalizer.denormalize(v_next), done, gamma=self.gamma, lam=0.95)
        # v_target_total = self.value_normalizer.denormalize(v_next) * self.gamma + reward
        v_target_total = self.value_normalizer.denormalize(v_value) + advantage
        self.value_normalizer.update(v_target_total)
        v_target_total = self.value_normalizer.normalize(v_target_total)

        # advantage = compute_gae(reward, v_value, v_next, done, gamma=self.gamma, lam=0.95)
        # # v_target_total = v_next * self.gamma + reward
        # v_target_total = v_value + advantage

        self.AC_training.train()

        for _ in pbar:
            loss = 0
            loss_kl = 0
            indices = np.random.randint(0, len(worker_state), size=batch_size)
            for j in range(batch_size):
                i = indices[j]
                img_temp, worker_state_temp, order_state_temp, order_num_temp, pool_order_temp, action_temp, p_log_old_temp, reward_temp, img_next_temp, worker_state_next_temp, order_state_next_temp, order_num_next_temp, pool_order_next_temp, action_next_temp = \
                img[i], worker_state[i], order_state[i], order_num[i], pool_order[i], action[i], p_log_old[i], reward[
                    i], img_next[i], worker_state_next[i], order_state_next[i], order_num_next[i], pool_order_next[i], \
                action_next[i]
                img_temp, worker_state_temp, order_state_temp, order_num_temp, pool_order_temp, action_temp = img_temp.to(
                    self.device), worker_state_temp.to(self.device), order_state_temp.to(
                    self.device), order_num_temp.to(
                    self.device), pool_order_temp.to(self.device), action_temp.to(self.device)

                x1, x2, x3 = norm(pool_order_temp, worker_state_temp, order_state_temp)
                img_temp = img_norm(img_temp)
                v, prob, prob_log = self.AC_training(img_temp, x1, x2, x3, order_num_temp)

                v_target = v_target_total[i]
                # loss_critic = (v-v_target.detach()) ** 2

                error_ori = v_target - v
                loss_ori = huber_loss(error_ori, 10.0)
                v_clip = v_value[i] + (v - v_value[i]).clamp(-0.2,0.2)
                error_clip=  v_target - v_clip
                loss_clip = huber_loss(error_clip, 10.0)
                loss_critic = torch.max(loss_ori, loss_clip)

                valid_indices = (action_temp != -1)
                if not valid_indices.any():
                    loss += loss_critic
                    continue

                # _, prob_target, prob_log_target = self.AC_target(img_temp, x1, x2, x3, order_num_temp)
                # kl = ppo_kl_loss(prob[valid_indices], prob_log[valid_indices], prob_target[valid_indices], prob_log_target[valid_indices])
                # # kl = ppo_kl_loss(prob[worker_state_temp[:, 4] == 0], prob_log[worker_state_temp[:, 4] == 0],
                # #                       prob_target[worker_state_temp[:, 4] == 0],
                # #                       prob_log_target[worker_state_temp[:, 4] == 0])
                # loss_kl += kl

                entropy_per_sample = -torch.sum(prob[valid_indices] * prob_log[valid_indices], dim=1)
                # entropy_per_sample = -torch.sum(prob[worker_state_temp[:,4]==0] * prob_log[worker_state_temp[:,4]==0], dim=1)
                loss_entropy =  - entropy_per_sample.mean()
                # loss_entropy = 0

                selected_elements = prob_log[valid_indices, action_temp[valid_indices]]
                prob_log = torch.sum(selected_elements)
                ratio  = torch.exp(prob_log-p_log_old_temp)
                surr1 = ratio * advantage[i].detach()
                surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage[i].detach()
                loss_actor = -torch.min(surr1, surr2)

                loss += loss_critic + (loss_actor + loss_entropy * 0.05) * self.actor_rate


            loss_kl /= batch_size

            # # Early stopping if KL explodes
            # if loss_kl > self.early_stop_kl * 1.5:
            #    print(f"Early stopping at kl={loss_kl:.5f}")
            #     break
            # # Adaptive KL coefficient adjustment
            # if loss_kl > self.target_kl * 1.5:
            #     self.beta = min(self.beta * self.beta_adjust_factor, self.beta_max)
            # elif loss_kl < self.target_kl / 1.5:
            #     self.beta = max(self.beta / self.beta_adjust_factor, self.beta_min)

            loss /= batch_size
            loss = loss + self.beta * loss_kl
            loss_list.append(loss.item())

            self.optim.zero_grad(set_to_none=True)  # 推荐使用 set_to_none=True 节省显存
            scaled_loss = self.scaler.scale(loss)  # 缩放 loss
            scaled_loss.backward()  # 反向传播（也可以写成 self.scaler.scale(loss).backward()）
            # ─────────────── 梯度 unscale ───────────────
            self.scaler.unscale_(self.optim)
            # # 检查 unscaled 后的真实梯度是否有 NaN 或 Inf
            has_invalid_grad = False
            for param in self.AC_training.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_invalid_grad = True
                        break
            if has_invalid_grad:
                print("NaN/Inf detected in unscaled gradients → skipping update")
                self.scaler.update()  # 重要：让 scaler 知道这次有溢出，下次缩小 scale
                continue  # 跳过本次参数更新
            # 梯度裁剪（现在操作的是真实梯度）
            torch.nn.utils.clip_grad_norm_(
                self.AC_training.parameters(),
                max_norm=1.0,  # 可根据你的任务调整：0.5～5.0 都很常见
                norm_type=2.0
            )
            # 更新参数
            self.scaler.step(self.optim)  # 内部还会再检查一次 inf/nan（双保险）
            self.scaler.update()  # 更新缩放因子（成功时尝试增大，溢出时已在上方处理）

        if len(loss_list) == 0:
            loss_list.append(0)
        torch.set_grad_enabled(False)
        # self.update_target(1.0)
        return np.mean(loss_list)

    def update(self, p_log, feedback_table, new_route_table ,new_route_time_table ,new_remaining_time_table ,new_total_travel_time_table, new_detour_table, reward, assignment_table, assignment_state, final_step=False, episode=1):
        # update each worker state parallely
        results = Parallel(n_jobs=self.njobs)(
            delayed(single_update)(self.observe_space[i], self.current_orders[i], self.current_order_num[i], self.travel_route[i], self.travel_time[i], feedback_table[i], new_route_table[i], new_route_time_table[i], new_remaining_time_table[i], new_total_travel_time_table[i], new_detour_table[i])
            for i in range(self.num))
        if self.is_train:
            assignment_table = [-1 if x is None else x for x in assignment_table]
            state = [self.observe_space.copy(),self.current_orders.copy(),self.current_order_num.copy(),assignment_state,self.img.copy()]

            assignment_table = np.array(assignment_table)
            valid_indices = (assignment_table != -1)
            selected_elements = p_log[valid_indices, assignment_table[valid_indices]]
            p_log = np.sum(selected_elements)
            assignment_table = assignment_table.tolist()

            action = [assignment_table,p_log]
            self.experience.append(state)
            self.experience.append(action)
            if len(self.experience) == 5:
                self.buffer.append(self.experience,episode)
                self.experience = [state, action, reward]
            else:
                self.experience.append(reward)
            if final_step and len(self.experience)>0:
                self.experience.append([None,None,None,None,None])
                self.experience.append(None)
                self.buffer.append(self.experience,episode)

        for i in range(len(results)):
            self.observe_space[i], self.current_orders[i], self.current_order_num[i], self.travel_route[i], \
            self.travel_time[i] = results[i][0], results[i][1], results[i][2], results[i][3], results[i][4]
            if results[i][5] is not None:
                self.Pass_Travel_Time.extend(results[i][5].tolist())
                self.Detour_Time.extend(results[i][6].tolist())

        # # update idle time
        self.observe_space[:, 6] += 1
        self.observe_space[self.current_order_num != 0, 6] = 0


def single_update(observe_space, current_orders, current_orders_num, current_travel_route, current_travel_time, feedback, new_route ,new_route_time, new_remaining_time, new_total_travel_time, new_detour_table):
    finished_order_time = None
    finished_order_detour = None
    current_orders_num = int(current_orders_num)

    if feedback is not None:
        new_order_state = feedback[0][3]
        pickup_time = feedback[2]

        # update state
        observe_space[0] = new_order_state[0]  # plat
        observe_space[1] = new_order_state[1]  # plon
        observe_space[2] -= 1  # remaining seat
        observe_space[3] = pickup_time  # pickup time
        observe_space[4] = 1  # update to picking up state
        current_travel_route, current_travel_time = new_route, new_route_time
        current_orders[:current_orders_num + 1, 2], current_orders[:current_orders_num + 1, 3], current_orders[:current_orders_num + 1, 4] = new_remaining_time, new_total_travel_time, new_detour_table
        current_orders[current_orders_num, 0], current_orders[current_orders_num, 1] = new_order_state[2], new_order_state[3]  # dlat,dlon (new orders)
        current_orders_num += 1

    # simulate 1 min
    step = 1  # 1min
    if observe_space[3] != 0:  # pick up
        if observe_space[3] > step:
            observe_space[3] -= step
            step = 0
        else:  # finish picking up
            step -= observe_space[3]
            observe_space[3] = 0
            if observe_space[2] != 0:  # have available seat
                observe_space[4] = 0 # update state to available

    if step > 0 and current_orders_num != 0:
        step_minute = step
        step = step * 60
        for i in range(len(current_travel_time)):
            if step >= current_travel_time[i]:
                step -= current_travel_time[i]
            else:
                current_travel_time[i] -= step
                current_travel_time = current_travel_time[i:]
                current_travel_route = current_travel_route[i:]
                observe_space[0], observe_space[1] = current_travel_route[0][1], current_travel_route[0][0]  # lat, lon
                break
            if i == len(current_travel_time) - 1:  # finish all orders
                observe_space[0], observe_space[1] = current_travel_route[-1][1], current_travel_route[-1][0]  # lat, lon
                current_travel_time = []
                current_travel_route = []
        current_orders[:current_orders_num, 2] -= step_minute  # update remaining time

        # delete finished orders
        drop_index = np.zeros(current_orders.shape[0])
        drop_index[:current_orders_num] = (current_orders[:current_orders_num, 2] <= 0)
        drop_num = np.sum(drop_index)
        if drop_num > 0:
            current_orders_num -= drop_num
            observe_space[2] += drop_num
            observe_space[4] = 0
            drop_index = drop_index.astype(bool)
            finished_orders = current_orders[drop_index]
            current_orders = current_orders[~drop_index]
            fill_matrix = np.zeros_like(finished_orders)
            current_orders = np.concatenate([current_orders, fill_matrix], axis=0)
            finished_order_time = finished_orders[:, 3]
            finished_order_detour = finished_orders[:, 4]

    return observe_space, current_orders, current_orders_num, current_travel_route, current_travel_time, finished_order_time, finished_order_detour

def ppo_kl_loss(
        prob: torch.Tensor,  # 当前策略 (new/policy) 的概率，形状 (n, m)
        prob_log: torch.Tensor,  # 当前策略的 log prob，形状 (n, m)
        prob_target: torch.Tensor,  # 旧策略 / target policy 的概率，形状 (n, m)
        prob_log_target: torch.Tensor  # 旧策略的 log prob，形状 (n, m)
) -> torch.Tensor:
    """
    返回 PPO-KL Penalty 中的 KL divergence 项（KL(old || new)）。
    在 PPO-Penalty 版本中，通常将此 KL 值乘以系数 β 后加入 loss（作为 penalty 项）。
    大多数实现（包括实际代码和论文约束）使用 KL(old || new)。

    输入假设：
    - 所有 tensor 都在同一设备（cpu/cuda）
    - prob 和 prob_target 是有效的概率分布（每行 sum ≈ 1.0）
    - prob_log ≈ torch.log(prob)，prob_log_target ≈ torch.log(prob_target)
    """
    # 数值稳定：可选 clip 避免极端值（生产环境推荐）
    prob_target = prob_target.clamp(min=1e-10)
    prob = prob.clamp(min=1e-10)

    # KL(old || new)
    kl_per_sample = (prob_target * (prob_log_target - prob_log)).sum(dim=-1)
    kl_mean = kl_per_sample.mean()

    return kl_mean