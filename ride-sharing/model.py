import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BartModel, BartConfig
import math
from peft import LoraConfig, get_peft_model, TaskType

class ZeroEmbedding(nn.Module):
    def forward(self, *args, **kwargs):
        return torch.tensor([0])

class MyBART(nn.Module):
    def __init__(self, hidden_dim=64,layer=3,head=4,max_len=20,lora=False,lora_rank=8,lora_alpha=16,lora_dropout=0.1):
        super().__init__()
        # model = BartModel.from_pretrained("facebook/bart-base")
        config = BartConfig(
            max_position_embeddings=max_len,
            encoder_layers=layer,
            encoder_ffn_dim=hidden_dim,
            encoder_attention_heads=head,
            decoder_layers=layer,
            decoder_ffn_dim=hidden_dim,
            decoder_attention_heads=head,
            encoder_layerdrop=0.0,
            decoder_layerdrop=0.0,
            activation_function="gelu",
            d_model=hidden_dim,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            is_encoder_decoder=True,
        )
        model = BartModel(config)

        del model.shared
        del model.encoder.embed_tokens
        del model.decoder.embed_tokens
        model.encoder.embed_positions = ZeroEmbedding()
        self.hidden_dim = model.config.d_model

        if lora:
            peft_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules="all-linear")
            self.peft_config = peft_config
            model = get_peft_model(model, peft_config)

        self.model = model
        self.lora = lora

    def forward(self, input_embeds, decoder_input_embeds, attention_mask=None, decoder_attention_mask=None):
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            decoder_inputs_embeds=decoder_input_embeds,
            decoder_attention_mask=decoder_attention_mask,
        )
        return outputs.encoder_last_hidden_state, outputs.last_hidden_state

    def encode(self, input_embeds, attention_mask=None):
        outputs = self.model.encoder(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state

    def decode(self, encoder_hidden_states, decoder_input_embeds, attention_mask=None, decoder_attention_mask=None):
        outputs = self.model.decoder(
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_input_embeds,
            attention_mask=decoder_attention_mask,
        )
        return outputs.last_hidden_state

class MLP(nn.Module):
    def __init__(self, layer_sizes = [64,64,64,1], arl = False, dropout = 0.0, bias = True):
        super().__init__()
        self.arl = arl
        if self.arl:
            self.attention = nn.Sequential(
                nn.Linear(layer_sizes[0],layer_sizes[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(layer_sizes[0],layer_sizes[0])
            )

        self.layer_sizes = layer_sizes
        if len(layer_sizes) < 2:
            raise ValueError()
        self.layers = nn.ModuleList()
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout(dropout)
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias = bias))

    def forward(self, x):
        if self.arl:
            x = x * self.attention(x)
        for layer in self.layers[:-1]:
            x = self.dropout(self.act(layer(x)))
        x = self.layers[-1](x)
        return x


class Self_Attention(nn.Module):
    def __init__(self, input_dim, da, r):
        super().__init__()
        self.ws1 = nn.Linear(input_dim, da, bias=False)
        self.ws2 = nn.Linear(da, r, bias=False)

    def forward(self, h):
        attn_mat = F.softmax(self.ws2(torch.tanh(self.ws1(h))), dim=1)
        attn_mat = attn_mat.permute(0, 2, 1)
        return attn_mat


class SelfCompression(nn.Module):
    def __init__(self, input_dim, output_dim, da, r):
        super().__init__()
        self.attn = Self_Attention(input_dim,da,r)
        self.mlp = MLP([input_dim*r, input_dim, output_dim])

    def forward(self, x):
        attn_mat = self.attn(x)
        y = torch.bmm(attn_mat, x)
        y = y.view(y.size()[0], -1)
        y = self.mlp(y)
        return y


class QK_Attention(nn.Module):
    def __init__(self, q_dims=64, k_dims=64, hidden_dims=64, head=1, dropout=0.0, method="mean", share = False):
        super().__init__()
        self.q_linear = nn.ModuleList()
        self.k_linear = nn.ModuleList()
        for i in range(int(head)):
            self.q_linear.append(MLP([q_dims,hidden_dims*2,hidden_dims*4], dropout=dropout))
            if not share:
                self.k_linear.append(MLP([k_dims,hidden_dims*2,hidden_dims*4], dropout=dropout))
        if share:
            self.k_linear = self.q_linear
        self.share = share
        self.head = head
        self.method = method

        self.softmax = nn.Softmax(dim=-1)

        if self.method != "mean":
            self.fuse_layer = MLP([head,head,1], dropout=dropout)

    def forward(self,q,k):
        attn_matrix = None

        for i in range(self.head):
            query=self.q_linear[i](q)
            if not self.share:
                key = self.k_linear[i](k)
                # key = self.softmax(key)
                key = key ** 2
                norms = torch.norm(key, dim=1, keepdim=True) + 1e-8
                key = key / norms
            else:
                key = query
            attn = torch.mm(query,key.T)

            if self.head == 1:
                return attn

            attn = attn.unsqueeze(-1)
            if attn_matrix is None:
                attn_matrix = attn
            else:
                attn_matrix = torch.concat([attn_matrix, attn],dim=-1)

        if self.method == "mean":
            attn_matrix = torch.mean(attn_matrix,dim=-1)
        else:
            attn_matrix = self.fuse_layer(attn_matrix)
            attn_matrix = attn_matrix.squeeze(-1)
        return attn_matrix


def init(module, weight_init, bias_init, gain=1.0):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

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


def get_sinusoidal_pos_emb(seq_len, d_model, device=torch.device("cpu")):
    position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() *
                         (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, seq_len, d_model)


class CMAT(nn.Module):
    def __init__(self, state_size=11, enroute_order_size=4, new_order_size=5, grid_size=14, grid_num=15, hidden_dim=64, consensus_len=5, finetune=None):
        super().__init__()

        self.transformer = MyBART(hidden_dim=hidden_dim, lora=False)


        # parameters
        self.grid_num = grid_num
        self.consensus_len = consensus_len
        self.hidden_dim = hidden_dim

        # encode worker information
        self.worker_encoder = MLP([state_size,hidden_dim,hidden_dim], arl=True)
        self.enroute_encoder = MLP([enroute_order_size,hidden_dim,hidden_dim],arl=True)
        self.enroute_config = BertConfig(max_position_embeddings=10, hidden_size=hidden_dim, intermediate_size=hidden_dim, num_hidden_layers=2, num_attention_heads=4)
        self.enroute_bert = BertModel(self.enroute_config)
        del self.enroute_bert.embeddings.word_embeddings
        self.enroute_compressor = SelfCompression(input_dim=hidden_dim, da=hidden_dim, r=4, output_dim=hidden_dim)
        self.sos = nn.Parameter(torch.randn([1, 1, hidden_dim]), requires_grad=True)
        self.pad = nn.Parameter(torch.randn([1, 1, hidden_dim]), requires_grad=True)
        self.fuse_encoder = MLP([hidden_dim*2,hidden_dim])

        # # encode map information
        # self.grid_encoder = MLP([grid_size,hidden_dim,hidden_dim],arl=True)
        # self.grid_config = BertConfig(max_position_embeddings=100, hidden_size=hidden_dim,intermediate_size=hidden_dim, num_hidden_layers=2, num_attention_heads=4, position_embedding_type="none")
        # self.grid_bert = BertModel(self.grid_config)
        # del self.grid_bert.embeddings.word_embeddings
        # del self.grid_bert.embeddings.position_embeddings
        # # ViT Part
        # self.recoverer = MLP([hidden_dim, hidden_dim, grid_size])
        # self.predictor = MLP([hidden_dim, hidden_dim, grid_size])
        # self.pad_consensus = nn.Parameter(torch.randn([1,self.transformer_hidden]), requires_grad=True)


        # encode order information
        self.order_encoder = MLP([new_order_size,hidden_dim,hidden_dim],arl=True)


        # Initial Consensus
        self.compressor = SelfCompression(input_dim=hidden_dim, da=hidden_dim, r=4, output_dim=hidden_dim)
        self.V_estimator = MLP([hidden_dim,hidden_dim,1])
        # Decoder
        # self.projector = nn.Sequential(nn.LayerNorm(hidden_dim),
        #                                init_(nn.Linear(hidden_dim, hidden_dim), activate=True), nn.GELU(),
        #                                nn.LayerNorm(hidden_dim),
        #                                init_(nn.Linear(hidden_dim, hidden_dim)))
        self.gate = nn.Sequential(nn.LayerNorm(hidden_dim),
                                  init_(nn.Linear(hidden_dim, hidden_dim), activate=True), nn.GELU(), nn.LayerNorm(hidden_dim),
                                  init_(nn.Linear(hidden_dim, 1)))
        # Utility Generator
        self.qk_attn = QK_Attention(hidden_dim*2,hidden_dim,hidden_dim,1)
        self.softmax = nn.Softmax(dim=1)

        if finetune is not None:
            for p in self.parameters():
                p.requires_grad = False
            for p in self.V_estimator.parameters():
                p.requires_grad = True
            if finetune == "action":
                for p in self.qk_attn.parameters():
                    p.requires_grad = True
            elif finetune == "consensus":
                for p in self.transformer.model.decoder.parameters():
                    p.requires_grad = True
            else:
                print("Not Valid Fine-tuning Mode")
                exit()

    def enroute_encode(self,x,valid_index=None):
        x = self.enroute_encoder(x)
        sos = self.sos.repeat(x.shape[0], 1, 1)
        x = torch.concat([sos,x],dim=1)
        device = x.device

        batch_size = len(valid_index)
        positions = torch.arange(4).unsqueeze(0).expand(batch_size, -1).to(device)
        valid_lengths = valid_index.unsqueeze(1).to(device)
        attention_mask = (positions <= valid_lengths).long()
        attention_mask = attention_mask

        attention_mask_expand = attention_mask.unsqueeze(-1)
        x = x * attention_mask_expand * self.pad * (1-attention_mask_expand)

        x = self.enroute_bert(inputs_embeds=x,attention_mask=attention_mask,output_hidden_states=False)
        x = x.last_hidden_state

        y = self.enroute_compressor(x)
        return y

    def encode(self, worker_state, enroute_order_state, order_num, new_order_state, grid_state):
        worker_state = self.worker_encoder(worker_state)
        enroute_state = self.enroute_encode(enroute_order_state,order_num)
        worker_state = self.fuse_encoder(torch.concat([worker_state,enroute_state],dim=-1))
        new_order_state = self.order_encoder(new_order_state)
        # grid_state = grid_state.unsqueeze(0)
        # grid_state = self.grid_encoder(grid_state)
        # grid_state = self.grid_bert(inputs_embeds=grid_state,output_hidden_states=False)
        # grid_state = grid_state.last_hidden_state
        # grid_state = grid_state.squeeze(0)

        # x0 = torch.concat([worker_state,new_order_state,grid_state],dim=0).unsqueeze(0)
        x0 = torch.concat([worker_state,new_order_state],dim=0).unsqueeze(0)
        x = self.transformer.encode(x0)
        # x = x + x0

        return x


    def forward(self, grid_state, new_order_state, worker_state, enroute_order_state, enroute_order_num, exploration_rate = 0):
            grid_state, new_order_state, worker_state, enroute_order_state, enroute_order_num = grid_state.float(), new_order_state.float(), worker_state.float(), enroute_order_state.float(), enroute_order_num.float()
            x_enc = self.encode(worker_state, enroute_order_state, enroute_order_num, new_order_state, grid_state)

            # V-value
            prefix = self.compressor(x_enc)
            v_value = self.V_estimator(prefix).squeeze(-1)

            # Action
            sos = prefix
            x_dec = torch.zeros([x_enc.shape[0], self.consensus_len, x_enc.shape[2]]).to(x_enc.device)
            x_dec[:, 0, :] = x_dec[:, 0, :] + sos
            for i in range(x_dec.shape[1]):

                x = x_dec
                x = self.transformer.decode(x_enc,x)
                # x = self.projector(x)

                if i != x_dec.shape[1] - 1:
                    x_dec = torch.concat([x_dec[:, :i + 1, :], x[:, i:-1, :]], dim=1)
                    x_dec[:,i+2:,:] = 0
                else:
                    x_dec = torch.concat([x_dec[:, :i + 1, :], x[:, i:, :]], dim=1)
                    x = x_dec

            weight = self.softmax(self.gate(x).squeeze(-1)).unsqueeze(-1)
            consensus = torch.sum(weight * x, dim=1, keepdim=True)
            consensus = consensus.repeat(1, x_enc.shape[1], 1)

            x_enc = torch.concat([x_enc,consensus],dim=-1).squeeze(0)
            worker_state = x_enc[:worker_state.shape[0]]
            # action_state = x_enc[worker_state.shape[0]:-self.grid_num]
            action_state = x_enc[worker_state.shape[0]:]
            utility = self.qk_attn(worker_state,action_state[:,:self.hidden_dim])
            logit = self.softmax(utility)+1e-8
            logit_log = torch.log(logit)
            if exploration_rate == 0:
                return v_value, logit, logit_log
            else:
                logit_temperature = self.softmax(utility / (1+exploration_rate)) + 1e-8
                logit_log_temperature = torch.log(logit_temperature)
                return v_value, logit, logit_log, logit_log_temperature

if __name__ == '__main__':
    model = CMAT(state_size=11, enroute_order_size=4, new_order_size=5, grid_size=14, grid_num=15, hidden_dim=64, consensus_len=5)
    print('Total parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    worker_state = torch.randn([10,11])
    enroute_state = torch.randn([10,3,4])
    order_num = torch.randint(0,3,(10,))
    new_order_state = torch.randn([15,5])
    grid_state = torch.randn([15,14])
    action = torch.randint(0,30,[10])
    reward = torch.randn([10])
    v_value, logit, logit_log = model(grid_state, new_order_state, worker_state, enroute_state, order_num)
    print(v_value.shape)
    print(logit.shape)
    print(logit_log.shape)