import torch
from torch import nn
import torch.nn.functional as F
from models.layers import get_all_indices, Embedding, MLP

class KnwlAttentiveRegLayer(nn.Module):

    def __init__(self, reg_type, knwl_param, knwl_matrix, nfield, ninfeat, nemb, noutfeat, nhead, dropout, alpha):
        super().__init__()
        # self.nhead = nhead  # attnreg, no nheads!!
       
        self.reg_type = reg_type
        self.knwl_param = knwl_param
        self.knwl_matrix = knwl_matrix                                           # F*F, including self-edge

        if self.reg_type = 1 or self.reg_type = 3 or self.reg_type = 5 or self.reg_type = 9:
            self.W_knwl = nn.Parameter(torch.zeros_like(self.knwl_matrix))       # F*F, weight matrix for knowledge, including self-edge
            nn.init.xavier_uniform_(self.W_knwl, gain=1.414)
        elif self.reg_type = 2 or self.reg_type = 4 or self.reg_type = 6 or self.reg_type = 10:
            self.w_knwl = nn.Parameter(torch.tensor(1.))                    # weight for knowledge
        
        self.W_Q = nn.Parameter(torch.zeros(ninfeat, nemb))       # E1*E, only this layer, attnreg
        nn.init.xavier_uniform_(self.W_Q, gain=1.414)

        self.W_Key = nn.Parameter(torch.zeros(ninfeat, nemb))       # E1*E, only this layer, attnreg
        nn.init.xavier_uniform_(self.W_Key, gain=1.414)

        self.values = nn.Parameter(torch.zeros(nfield, nfield))       # F*F, including self-edge, attnreg
        nn.init.xavier_uniform_(self.values, gain=1.414)

        self.W = nn.Parameter(torch.zeros(size=(nfield*nfield, ninfeat, noutfeat)))  # (FXF)*E1*E2, assume all connected ..., including self-edge, attnreg
        print ("self.W size: ", self.W.size())
        for i in range(self.W.shape[0]):
            # print ("self.W[i] size: ", self.W[i].size())
            nn.init.xavier_uniform_(self.W[i].data, gain=1.414)

        # for _ in range(nhead):
            # self.W.append(nn.Parameter(torch.zeros(size=(ninfeat, noutfeat))))
            # nn.init.xavier_uniform_(self.W[-1].data, gain=1.414)
            # self.a.append(nn.Linear(2*noutfeat, 1, bias=False))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, knwl_metric, normalization_coefficient, verbose_flag, adj):
        '''
        :param x:       FloatTensor B*F*E1
        :param normalization_coefficient: normalization coefficient for regularization
        :verbose_flag:  whether to print
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
        
        ### x * w --> for neighbouring nodes
        h_list = []
        for i in range(self.W.shape[0]):                                          # FxF
            if verbose_flag and i == 50:
                print ("i of h_list: ", i)
                print ("xj[:, i] norm: ", torch.norm(xj[:, i]))
                print ("self.W[i] norm: ", torch.norm(self.W[i]))
            h_list.append(torch.einsum('bfi,io->bfo', xj[:, i].view(xj.shape[0], 1, -1), self.W[i]))        # B*(1)*E2, attnreg 1: x[:, i] shape ??, attnreg 2: torch.cat(customized_h_list, dim=1) ??
        h = torch.cat(h_list, dim=1)                                              # B*(FxF)*E2
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
        # attn = torch.einsum('bxy,xy->bxy', e.view(-1, nfield, nfield), mask)    # B*F*F
        #### different reg types -- begin
        if self.reg_type == 1:
            knwl_mul = torch.einsum('xy,xy->xy', self.W_knwl, self.knwl_matrix)   # F*F
            print ("knwl_mul shape: ", knwl_mul.shape)
            attn = torch.einsum('bxy,xy->bxy', attn, knwl_mul)                    # B*F*F
        elif self.reg_type == 2:
            knwl_mul = self.w_knwl * self.knwl_matrix
            print ("knwl_mul shape: ", knwl_mul.shape)
            attn = torch.einsum('bxy,xy->bxy', attn, knwl_mul)                    # B*F*F
        elif self.reg_type == 3:
            knwl_add = torch.einsum('xy,xy->xy', self.W_knwl, self.knwl_matrix)   # F*F
            print ("knwl_add shape: ", knwl_add.shape)
            attn = attn + knwl_add                                                # B*F*F, broadcasting
            print ("attn broadcasting shape: ", attn.shape)
        elif self.reg_type == 4:
            knwl_add = self.w_knwl * self.knwl_matrix
            print ("knwl_add shape: ", knwl_add.shape)
            attn = attn + knwl_add                                                # B*F*F, broadcasting
            print ("attn broadcasting shape: ", attn.shape)
        else:
            pass
        #### different reg types -- end
        # attn = self.dropout(F.softmax(attn.view(-1,nfield,nfield), dim=-1))        # B*F*F
        # if self.reg_type == 7:  # KL-Divergence
        #     attn = F.log_softmax(attn.view(-1,nfield,nfield), dim=-1)
        # else:
        #     attn = F.softmax(attn.view(-1,nfield,nfield), dim=-1)
        # change softmax here ....
        if self.reg_type == 9 or self.reg_type == 10 or self.reg_type == 11:
            if self.reg_type == 9:
                knwl_mul = torch.einsum('xy,xy->xy', self.W_knwl, self.knwl_matrix)   # F*F
                print ("knwl_mul shape: ", knwl_mul.shape)
            elif self.reg_type == 10:
                knwl_mul = self.w_knwl * self.knwl_matrix
                print ("knwl_mul shape: ", knwl_mul.shape)
            else:  # no knwl weights
                knwl_mul = 1.0 * self.knwl_matrix  # had better to be a new one ...
            ### not using function because not sure whether weights can be passed into ...
            maxes = torch.max(attn.view(-1,nfield,nfield), -1, keepdim=True)[0]  # returns (data, indices)
            print ("maxes shape: ", maxes.shape)
            print ("maxes: ", maxes)
            x_exp = torch.exp(x-maxes)
            print ("x_exp shape: ", x_exp)
            x_exp = (1.0 - self.knwl_param) * x_exp + self.knwl_param * knwl_mul  # broadcasting !!!
            print ("x_exp broadcasting shape: ", x_exp.shape)
            x_exp_sum = torch.sum(x_exp, -1, keepdim=True)
            attn = x_exp/x_exp_sum
        else:
            attn = F.softmax(attn.view(-1,nfield,nfield), dim=-1)
        print ("attn shape: ", attn.shape)
        layer_knwl_loss = None
        if self.reg_type == 7 or self.reg_type == 8:
            print ("knwl_metric: ", knwl_metric)
            knwl_matrix_list = []
            for i in range(attn.shape[0]):
                knwl_matrix_list.append(self.knwl_matrix)
            knwl_matrix_concat = torch.cat(h_prime_list, dim=0)                  # (B*F)*F
            print ("knwl_matrix_concat shape: ", knwl_matrix_concat.shape)
            if self.reg_type == 7:  # KL-Divergence
                layer_knwl_loss = knwl_metric(torch.log(attn.view(-1,nfield)), knwl_matrix_concat) * nfield # this one log + needs to multiply by nfield !!
            else:
                layer_knwl_loss = knwl_metric(attn.view(-1,nfield), knwl_matrix_concat) * nfield # this one needs to multiply by nfield !!
        attn = self.dropout(attn)
        attn = 1.0 / attn                                                         # B*F*F --> inverse
        # reg_strength_values = torch.exp(self.values)  # option 1: exp
        reg_strength_values = torch.abs(self.values)  # option 2: abs
        # print ("reg_strength_values: ", reg_strength_values)
        reg_strength = torch.einsum('bxy,xy->bxy', attn, reg_strength_values)             # B*F*F
        # print ("einsum reg_strength shape: ", reg_strength.shape)
        # reg_strength = reg_strength.view(reg_strength.shape[0], -1)               # B*(FxF)
        reg_strength = torch.mean(reg_strength.view(reg_strength.shape[0], -1), dim=0) # B*(FxF) --> (FxF) average over all samples in this batch ...
        # print ("in reg normalization_coefficient: ", normalization_coefficient)
        reg_strength = reg_strength / normalization_coefficient
        # print ("reg_strength: ", reg_strength)
        # print ("after mean reg_strength shape: ", reg_strength.shape)
        layer_reg_loss = None
        for i in range(reg_strength.shape[0]):
            # print ("reg_strength.shape[0]: ", reg_strength.shape[0])
            # print ("self.W[i].norm(2): ", self.W[i].norm(2))
            if layer_reg_loss is None:
                layer_reg_loss = (reg_strength[i] * self.W[i].norm(2) * self.W[i].norm(2)) 
            else:
                layer_reg_loss = layer_reg_loss + (reg_strength[i] * self.W[i].norm(2) * self.W[i].norm(2))          # can this be used for derivatives ??? can this be faster ???

        return torch.cat(h_prime_list, dim=1), layer_reg_loss, layer_knwl_loss                                                          # B*F*(E2)
        # return h_prime                                                           # B*F*E2

# Model:  Knowledge Attentive Regularization Model
class KnwlAttentiveRegModel(nn.Module):
    def __init__(self, reg_type, knwl_matrix, nfield, nfeat, nemb, gat_layers, gat_hid, mlp_layers, mlp_hid, dropout, alpha=0.2, nhead=8):
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)

        self.reg_type = reg_type
        self.knwl_matrix = knwl_matrix
        self.gat_layers = gat_layers
        self.gats = torch.nn.ModuleList()
        ninfeat = nemb
        for _ in range(gat_layers):
            self.gats.append(KnwlAttentiveRegLayer(self.reg_type, self.knwl_matrix, nfield, ninfeat, nemb, gat_hid, nhead, dropout, alpha))  # attnreg, pass in nfield
            # ninfeat = nhead*gat_hid  # attnreg no attention heads
            ninfeat = gat_hid

        self.dropout = nn.Dropout(p=dropout)
        self.affine = MLP(nfield*ninfeat, mlp_layers, mlp_hid, dropout)

    def forward(self, x, knwl_metric, normalization_coefficient, verbose_flag, adj=None):
        """
        :param x:   {'ids': LongTensor B*F, 'vals': FloatTensor B*F}
        :param normalization_coefficient: normalization coefficient for regularization
        :verbose_flag: whether to print
        :param adj:     FloatTensor F*F, default fully connected
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        reg_loss = None
        knwl_loss = None
        h = self.embedding(x)                                                   # B*F*E
        if verbose_flag:
            print ("entering model x norm: ", torch.norm(h))
            print ("entering model torch.norm(self.gats[0].W[0]) norm: ", torch.norm(self.gats[0].W[0]))
        # print ("h size: ", h.size())
        if adj is None:
            adj = torch.ones((h.size(0), h.size(1), h.size(1)), dtype=h.dtype, device=h.device)  # B*F*F attnreg
        for l in range(self.gat_layers):
            h, layer_reg_loss, layer_knwl_loss = self.gats[l](h, knwl_metric, normalization_coefficient, verbose_flag, adj)                                         # B*F*(gat_hid)
            # print ("h shape: ", h.shape)
            if verbose_flag:
                print ("after aggregating neighbouring nodes h norm: ", torch.norm(h))
            h = F.elu(self.dropout(h))
            if verbose_flag:
                print ("after elu and dropout h norm: ", torch.norm(h))
            if reg_loss is None:
                reg_loss = layer_reg_loss
            else:
                reg_loss = reg_loss + layer_reg_loss
            if self.reg_type == 7 or self.reg_type == 8:
                if knwl_loss is None:
                    knwl_loss = layer_knwl_loss
                else:
                    knwl_loss = knwl_loss + layer_knwl_loss 

        y = self.affine(h.view(h.size(0), -1))                                  # B*1
        return y.squeeze(1), reg_loss, knwl_loss                                           # B
