import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical
from mat.algorithms.utils.util import check, init
from mat.algorithms.utils.transformer_act import discrete_autoregreesive_act
from mat.algorithms.utils.transformer_act import discrete_parallel_act
from mat.algorithms.utils.transformer_act import continuous_autoregreesive_act
from mat.algorithms.utils.transformer_act import continuous_parallel_act
from torch.distributions import Categorical, Normal


class SelfCompression(nn.Module):
    def __init__(self, input_dim, output_dim, da, r):
        super().__init__()
        self.ws1 = nn.Linear(input_dim, da, bias=False)
        self.ws2 = nn.Linear(da, r, bias=False)
        self.ws3 = nn.Sequential(
            nn.LayerNorm(input_dim * r),
            init_(nn.Linear(input_dim * r, output_dim), activate=True), nn.GELU(), nn.LayerNorm(output_dim),
            init_(nn.Linear(output_dim, output_dim))
            # nn.LayerNorm(output_dim)
        )

    def forward(self, h):
        attn_mat = F.softmax(self.ws2(torch.tanh(self.ws1(h))), dim=1)
        attn_mat = attn_mat.permute(0, 2, 1)

        y = torch.bmm(attn_mat, h)
        y = y.view(y.size()[0], -1)
        y = self.ws3(y)
        return y


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
                             .view(1, 1, n_agent + 1, n_agent + 1))

        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()
        B, L1, D = key.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L1, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L1, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x):
        x = self.ln2(x)
        x = self.ln1(x + self.attn(x, x, x))
        # x = self.ln2(x + self.mlp(x))
        x = x + self.mlp(x)
        return x


class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        # self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, rep_enc):
        x = self.ln3(x)
        x = self.ln1(x + self.attn1(x, x, x))
        # x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln2(x + self.attn2(key=rep_enc, value=rep_enc, query=x))
        # x = self.ln3(x + self.mlp(x))
        x = x + self.mlp(x)
        return x


class Encoder(nn.Module):

    def __init__(self, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state):
        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state
        # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))

        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim),
                                           init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])

        self.compressor = SelfCompression(n_embd, n_embd, n_embd, 4)
        self.head = nn.Sequential(nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, 1)))

    def forward(self, state, obs):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        if self.encode_state:
            state_embeddings = self.state_encoder(state)
            x = state_embeddings
        else:
            obs_embeddings = self.obs_encoder(obs)
            x = obs_embeddings

        # rep = self.blocks(self.ln(x))
        rep = self.blocks(x)

        prefix = self.compressor(rep)
        # prefix = None

        v_loc = self.head(prefix)
        v_loc = v_loc.unsqueeze(-1).repeat(1, obs.shape[1], 1)
        # v_loc = self.head(rep)

        return v_loc, self.ln(rep), prefix


def get_sinusoidal_pos_emb(seq_len, d_model, device=torch.device("cpu")):
    position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() *
                         (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, seq_len, d_model)


class Decoder(nn.Module):

    def __init__(self, obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type

        if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            # log_std = torch.zeros(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
            # self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

        if self.dec_actor:
            print("Don't support dec_actor.")
            exit()
            # if self.share_actor:
            #     print("mac_dec!!!!!")
            #     self.mlp = nn.Sequential(nn.LayerNorm(obs_dim),
            #                              init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
            #                              init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
            #                              init_(nn.Linear(n_embd, action_dim)))
            # else:
            #     self.mlp = nn.ModuleList()
            #     for n in range(n_agent):
            #         actor = nn.Sequential(nn.LayerNorm(obs_dim),
            #                               init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
            #                               init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
            #                               init_(nn.Linear(n_embd, action_dim)))
            #         self.mlp.append(actor)
        else:
            # # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))
            # if action_type == 'Discrete':
            #     self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True),
            #                                         nn.GELU())
            # else:
            #     self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embd), activate=True), nn.GELU())
            # self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
            #                                  init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())
            # self.ln = nn.LayerNorm(n_embd)
            # self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
            # self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
            #                           init_(nn.Linear(n_embd, action_dim)))

            self.sos = nn.Parameter(torch.randn([1, n_embd]), requires_grad=True)

            # self.compressor = SelfCompression(n_embd, n_embd, n_embd, 4)
            self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])

            self.projector = nn.Sequential(nn.LayerNorm(n_embd),
                                           init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(),
                                           nn.LayerNorm(n_embd),
                                           init_(nn.Linear(n_embd, n_embd)))
            self.gate = nn.Sequential(nn.LayerNorm(n_embd),
                                      init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                      init_(nn.Linear(n_embd, 1)))
            self.softmax = nn.Softmax(dim=1)

            # self.head = nn.ModuleList()
            # for n in range(n_agent):
            #     actor = nn.Sequential(nn.LayerNorm(n_embd),
            #                           init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
            #                           init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
            #                           init_(nn.Linear(n_embd, action_dim)))
            #     self.head.append(actor)
            self.head = nn.Sequential(
                nn.LayerNorm(n_embd * 2),
                init_(nn.Linear(n_embd * 2, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                init_(nn.Linear(n_embd, action_dim)))
            self.pos_emb = get_sinusoidal_pos_emb(n_agent, n_embd)

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    # state, action, and return
    def forward(self, action, obs_rep, obs, prefix):
        self.pos_emb = self.pos_emb.to(obs_rep.device)

        # action: (batch, n_agent, action_dim), one-hot/logits?
        # obs_rep: (batch, n_agent, n_embd)
        if self.dec_actor:
            print("Don't support dec_actor.")
            exit()
            # if self.share_actor:
            #     logit = self.mlp(obs)
            # else:
            #     logit = []
            #     for n in range(len(self.mlp)):
            #         logit_n = self.mlp[n](obs[:, n, :])
            #         logit.append(logit_n)
            #     logit = torch.stack(logit, dim=1)
        else:
            # action_embeddings = self.action_encoder(action)
            # x = self.ln(action_embeddings)
            # for block in self.blocks:
            #     x = block(x, obs_rep)
            # logit = self.head(x)

            # sos = self.compressor(obs_rep)
            sos = prefix
            # sos = self.sos
            x_dec = torch.zeros_like(obs_rep)  # TODO: change the length
            x_dec[:, 0, :] = x_dec[:, 0, :] + sos
            x_dec = x_dec + self.pos_emb

            for i in range(x_dec.shape[1]):
                x = x_dec
                for block in self.blocks:
                    x = block(x, obs_rep)
                x_dec = x_dec - self.pos_emb

                x = self.projector(x)

                if i != x_dec.shape[1] - 1:
                    x_dec = torch.concat([x_dec[:, :i + 1, :], x[:, i:-1, :]], dim=1)
                    x_dec = x_dec + self.pos_emb
                else:
                    x_dec = torch.concat([x_dec[:, :i + 1, :], x[:, i:, :]], dim=1)
                    x = x_dec

            # logit = []
            # for n in range(len(self.head)):
            #     logit_n = self.head[n](x[:, -1, :])
            #     logit.append(logit_n)
            # logit = torch.stack(logit, dim=1)

            weight = self.softmax(self.gate(x).squeeze(-1)).unsqueeze(-1)
            consensus = torch.sum(weight*x,dim=1,keepdim=True)

            consensus = consensus.repeat(1, obs_rep.shape[1], 1)
            logit = self.head(torch.concat([obs_rep, consensus], dim=-1))

        return logit


class ConsensusMultiAgentTransformer(nn.Module):

    def __init__(self, state_dim, obs_dim, action_dim, n_agent,
                 n_block, n_embd, n_head, encode_state=False, device=torch.device("cpu"),
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(ConsensusMultiAgentTransformer, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device

        # state unused
        state_dim = 37

        self.encoder = Encoder(state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state)
        self.decoder = Decoder(obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                               self.action_type, dec_actor=dec_actor, share_actor=share_actor)
        self.to(device)

    def zero_std(self):
        if self.action_type != 'Discrete':
            self.decoder.zero_std(self.device)

    def forward(self, state, obs, action, available_actions=None):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)

        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(state)[0]
        v_loc, obs_rep, prefix = self.encoder(state, obs)

        # if self.action_type == 'Discrete':
        #     action = action.long()
        #     action_log, entropy = discrete_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
        #                                                 self.n_agent, self.action_dim, self.tpdv, available_actions)
        # else:
        #     action_log, entropy = continuous_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
        #                                                   self.n_agent, self.action_dim, self.tpdv)

        if self.action_type == 'Discrete':
            logit = self.decoder(None, obs_rep, None, prefix)
            if available_actions is not None:
                logit[available_actions == 0] = -1e10
            distri = Categorical(logits=logit)
            action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
            entropy = distri.entropy().unsqueeze(-1)
        else:
            act_mean = self.decoder(None, obs_rep, None, prefix)
            action_std = torch.sigmoid(self.decoder.log_std) * 0.5
            distri = Normal(act_mean, action_std)
            action_log = distri.log_prob(action)
            entropy = distri.entropy()

        return action_log, v_loc, entropy

    def get_actions(self, state, obs, available_actions=None, deterministic=False):
        # state unused
        ori_shape = np.shape(obs)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        # batch_size = np.shape(obs)[0]
        v_loc, obs_rep, prefix = self.encoder(state, obs)

        # if self.action_type == "Discrete":
        #     output_action, output_action_log = discrete_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
        #                                                                    self.n_agent, self.action_dim, self.tpdv,
        #                                                                    available_actions, deterministic)
        # else:
        #     output_action, output_action_log = continuous_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
        #                                                                      self.n_agent, self.action_dim, self.tpdv,
        #                                                                      deterministic)

        if self.action_type == 'Discrete':
            logit = self.decoder(None, obs_rep, None, prefix)
            if available_actions is not None:
                logit[available_actions == 0] = -1e10

            distri = Categorical(logits=logit)
            action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
            action_log = distri.log_prob(action)
            output_action, output_action_log = action.unsqueeze(-1), action_log.unsqueeze(-1)
        else:
            act_mean = self.decoder(None, obs_rep, None, prefix)
            action_std = torch.sigmoid(self.decoder.log_std) * 0.5
            distri = Normal(act_mean, action_std)
            action = act_mean if deterministic else distri.sample()
            action_log = distri.log_prob(action)
            output_action, output_action_log = action.unsqueeze(-1), action_log.unsqueeze(-1)

        return output_action, output_action_log, v_loc

    def get_values(self, state, obs, available_actions=None):
        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        v_tot, obs_rep, prefix = self.encoder(state, obs)
        return v_tot



