import torch
from torch import nn
import torch.nn.functional as F
from models.layers import get_all_indices, Embedding, MLP

class AttentiveRegLayer(nn.Module):

    def __init__(self, nfield, ninfeat, nemb, noutfeat, nhead, dropout, alpha):
        super().__init__()
        # self.nhead = nhead  # attnreg, no nheads!!
        
        self.W_Q = nn.Parameter(torch.zeros(ninfeat, nemb))       # E1*E, only this layer, attnreg
        nn.init.xavier_uniform_(self.W_Q, gain=1.414)

        self.W_Key = nn.Parameter(torch.zeros(ninfeat, nemb))       # E1*E, only this layer, attnreg
        nn.init.xavier_uniform_(self.W_Key, gain=1.414)

        self.values = nn.Parameter(torch.zeros(nfield, nfield))       # F*F, including self-edge, attnreg
        nn.init.xavier_uniform_(self.values, gain=1.414)

        self.W = nn.Parameter(torch.zeros(size=(nfield*nfield, ninfeat, noutfeat)))  # (FXF)*E1*E2, assume all connected ..., including self-edge, attnreg
        print ("self.W size: ", self.W.size())
        # print ("before init self.W: ", self.W)
        for i in range(self.W.shape[0]):
            # print ("self.W[i] size: ", self.W[i].size())
            nn.init.xavier_uniform_(self.W[i].data, gain=1.414)
        # print ("after init self.W: ", self.W)

        # for _ in range(nhead):
            # self.W.append(nn.Parameter(torch.zeros(size=(ninfeat, noutfeat))))
            # nn.init.xavier_uniform_(self.W[-1].data, gain=1.414)
            # self.a.append(nn.Linear(2*noutfeat, 1, bias=False))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, normalization_coefficient, reg_lambda, reg_method, verbose_flag, adj):
        '''
        :param x:       FloatTensor B*F*E1
        :param normalization_coefficient: normalization coefficient for regularization
        :param reg_lambda: regularization lambda for controlling regularization strength
        :param reg_method: regularization method for calculating regularization strength
        :verbose_flag: whether to print
        :param adj:     FloatTensor F*F
        :return:        FloatTensor B*F*E2
        '''
        nfield = x.size(1)
        zero_vec = -9e15 * torch.ones_like(adj)
        mask = torch.where(adj > 0, adj, zero_vec)                               # B*F*F, all ones, changed, including batches B in the first dimension, attnreg
        mask = 1/nfield * mask                                                   # this one is intialized to 1/nfield, attnreg, actually equals to attn for GAT, used for summing neighbours!!
        # print ("mask: ", mask)

        vi_indices, vj_indices = get_all_indices(nfield)
        # h = torch.einsum('bfi,io->bfo', x, self.W[head])                        # B*F*E2
        # vi, vj = h[:, vi_indices], h[:, vj_indices]
        # print ("vi_indices: ", vi_indices)
        # print ("vj_indices: ", vj_indices)
        xi, xj = x[:, vi_indices], x[:, vj_indices]                               # xj: B*(FxF)*(E1): [0,1,2,3,4,5..,9,0,1,2,3,4,5..,9,0,1,2,3,4,5..,9,0,1,2,3,4,5..,9]
        # print ("xi: ", xi)
        # print ("xj: ", xj)

        ### x * w --> for neighbouring nodes
        h_list = []
        for i in range(self.W.shape[0]):                                          # FxF
            if verbose_flag and i == 50:
                print ("i of h_list: ", i)
                print ("xj[:, i] norm: ", torch.norm(xj[:, i]))
                print ("self.W[i] norm: ", torch.norm(self.W[i]))
            h_list.append(torch.einsum('bfi,io->bfo', xj[:, i].view(xj.shape[0], 1, -1), self.W[i]))        # B*(1)*E2, attnreg 1: x[:, i] shape ??, attnreg 2: torch.cat(customized_h_list, dim=1) ??
        h = torch.cat(h_list, dim=1)                                              # B*(FxF)*E2
        # print ("h: ", h)
        if verbose_flag:
            print ("neighbouring nodes h norm: ", torch.norm(h))

        ### using mask for aggregating neighbouring nodes for central nodes
        h_prime_list = []
        for i in range(mask.shape[1]):
            h_prime = torch.einsum('bxy,bye->bxe', mask[:,i].view(mask.shape[0],1,-1), h[:,i*nfield:(i+1)*nfield])                           # B*1*E2
            h_prime_list.append(h_prime)
        
        ### attentive regularization
        query = torch.einsum('bfi,io->bfo', xi, self.W_Q)                         # B*(FxF)*E2
        key = torch.einsum('bfi,io->bfo', xj, self.W_Key)                         # B*(FxF)*E2
        attn = torch.einsum('bfo,bfo->bf', query, key)                            # B*(FxF)*1
        attn = self.dropout(F.softmax(attn.view(-1,nfield,nfield), dim=-1))       # B*F*F
        # print ("original attn: ", attn)
        # print ("original reg_lambda: ", reg_lambda)
        # print ("original normalization_coefficient: ", normalization_coefficient)
        # print ("original attn.shape[0]: ", attn.shape[0])
        reg_lambda = reg_lambda * normalization_coefficient                       # to offset the normalization_coefficient and batch size
        if reg_method == -1:  # this one still has the values
            attn = 1.0 / attn                                                     # B*F*F --> inverse
            # reg_strength_values = torch.exp(self.values)  # option 1: exp
            reg_strength_values = torch.abs(self.values)  # option 2: abs
            # print ("reg_strength_values: ", reg_strength_values)
            reg_strength = torch.einsum('bxy,xy->bxy', attn, reg_strength_values)             # B*F*F
            # print ("reg_strength shape before mean: ", reg_strength.shape)
            # print ("einsum reg_strength shape: ", reg_strength.shape)
            # reg_strength = reg_strength.view(reg_strength.shape[0], -1)               # B*(FxF)
            reg_strength = torch.mean(reg_strength.view(reg_strength.shape[0], -1), dim=0) # B*(FxF) --> (FxF) average over all samples in this batch ...
            # print ("in reg normalization_coefficient: ", normalization_coefficient)
        elif reg_method == 0:  # 1/attn
            # print ("1.0 / attn: ", 1.0 / attn)
            reg_strength = reg_lambda * (1.0 / attn)  # B*F*F, will have 1.0 / 0.0 ??? becaue of dropout ????
            # print ("reg_strength shape before mean: ", reg_strength.shape)
            # print ("reg_strength after reg_lambda * (1.0 / attn): ", reg_lambda * (1.0 / attn))
            reg_strength = torch.mean(reg_strength.view(reg_strength.shape[0], -1), dim=0) # B*(FxF) --> (FxF) average over all samples in this batch ...
            # print ("reg_strength after mean: ", reg_strength)
        elif reg_method == 1:  # exp(-attn)
            reg_strength = reg_lambda * torch.exp(-1.0 * attn)  # B*F*F, will have 1.0 / 0.0 ??? becaue of dropout ????
            # print ("reg_strength shape before mean: ", reg_strength.shape)
            reg_strength = torch.mean(reg_strength.view(reg_strength.shape[0], -1), dim=0) # B*(FxF) --> (FxF) average over all samples in this batch ...
        elif reg_method == 2:  # 1-attn
            reg_strength = reg_lambda * (1.0 - attn)  # B*F*F, will have 1.0 / 0.0 ??? becaue of dropout ????
            # print ("reg_strength shape before mean: ", reg_strength.shape)
            reg_strength = torch.mean(reg_strength.view(reg_strength.shape[0], -1), dim=0) # B*(FxF) --> (FxF) average over all samples in this batch ...
        elif reg_method == 3:  # -log(attn)
            reg_strength = -1.0 * reg_lambda * torch.log(attn)  # B*F*F, will have 1.0 / 0.0 ??? becaue of dropout ????
            # print ("reg_strength shape before mean: ", reg_strength.shape)
            reg_strength = torch.mean(reg_strength.view(reg_strength.shape[0], -1), dim=0) # B*(FxF) --> (FxF) average over all samples in this batch ...
        elif reg_method == 4:  # 1/(1+attn)
            reg_strength = reg_lambda * (1.0 / (1.0 + attn))  # B*F*F, will have 1.0 / 0.0 ??? becaue of dropout ????
            # print ("reg_strength shape before mean: ", reg_strength.shape)
            reg_strength = torch.mean(reg_strength.view(reg_strength.shape[0], -1), dim=0) # B*(FxF) --> (FxF) average over all samples in this batch ...
        else:
            print ("No reg_method specified!!!!!")
        reg_strength = reg_strength / normalization_coefficient
        # print ("reg_strength final shape: ", reg_strength.shape)
        if verbose_flag:
            print ("reg_strength: ", reg_strength)
        # print ("after mean reg_strength shape: ", reg_strength.shape)
        layer_reg_loss = None
        for i in range(reg_strength.shape[0]):
            # print ("reg_strength.shape[0]: ", reg_strength.shape[0])
            # print ("self.W[i].norm(2): ", self.W[i].norm(2))
            if layer_reg_loss is None:
                layer_reg_loss = (reg_strength[i] * self.W[i].norm(2) * self.W[i].norm(2)) 
            else:
                layer_reg_loss = layer_reg_loss + (reg_strength[i] * self.W[i].norm(2) * self.W[i].norm(2))          # can this be used for derivatives ??? can this be faster ???

        return torch.cat(h_prime_list, dim=1), layer_reg_loss                                                          # B*F*(E2)
        # return h_prime                                                           # B*F*E2

# Model:  Attentive Regularization Model
class AttentiveRegModel(nn.Module):
    def __init__(self, nfield, nfeat, nemb, gat_layers, gat_hid, mlp_layers, mlp_hid, dropout, alpha=0.2, nhead=8):
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)

        self.gat_layers = gat_layers
        self.gats = torch.nn.ModuleList()
        ninfeat = nemb
        for _ in range(gat_layers):
            self.gats.append(AttentiveRegLayer(nfield, ninfeat, nemb, gat_hid, nhead, dropout, alpha))  # attnreg, pass in nfield
            # ninfeat = nhead*gat_hid  # attnreg no attention heads
            ninfeat = gat_hid

        self.dropout = nn.Dropout(p=dropout)
        self.affine = MLP(nfield*ninfeat, mlp_layers, mlp_hid, dropout)

    def forward(self, x, normalization_coefficient, reg_lambda, reg_method, verbose_flag, adj=None):
        """
        :param x:   {'ids': LongTensor B*F, 'vals': FloatTensor B*F}
        :param normalization_coefficient: normalization coefficient for regularization
        :param reg_lambda: regularization lambda for controlloing strength of regularization
        :param reg_method: regularization method for calculating regularization strength
        :verbose_flag: whether to print
        :param adj:     FloatTensor F*F, default fully connected
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        reg_loss = None
        # print ("x: ", x)
        h = self.embedding(x)                                                   # B*F*E
        # print ("h: ", h)
        if verbose_flag:
            print ("entering model x norm: ", torch.norm(h))
            print ("entering model torch.norm(self.gats[0].W[0]) norm: ", torch.norm(self.gats[0].W[0]))
        # print ("h size: ", h.size())
        if adj is None:
            adj = torch.ones((h.size(0), h.size(1), h.size(1)), dtype=h.dtype, device=h.device)  # B*F*F attnreg
        for l in range(self.gat_layers):
            h, layer_reg_loss = self.gats[l](h, normalization_coefficient, reg_lambda, reg_method, verbose_flag, adj)                                         # B*F*(gat_hid)
            # print ("h shape: ", h.shape)
            # print ("h: ", h)
            if verbose_flag:
                print ("after aggregating neighbouring nodes h norm: ", torch.norm(h))
            h = F.elu(self.dropout(h))
            # print ("h: ", h)
            if verbose_flag:
                print ("after elu and dropout h norm: ", torch.norm(h))
            if reg_loss is None:
                reg_loss = layer_reg_loss
            else:
                reg_loss = reg_loss + layer_reg_loss

        y = self.affine(h.view(h.size(0), -1))                                  # B*1
        # print ("y: ", y)
        return y.squeeze(1), reg_loss                                           # B
