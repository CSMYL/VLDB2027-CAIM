import torch
from torch import nn
import torch.nn.functional as F
from models.layers import get_all_indices, Embedding, MLP

class AttentiveNoRegLayer(nn.Module):

    def __init__(self, nfield, ninfeat, nemb, noutfeat, nhead, dropout, alpha):
        super().__init__()

        self.W = nn.Parameter(torch.zeros(size=(nfield*nfield, ninfeat, noutfeat)))  # (FXF)*E1*E2, assume all connected ..., including self-edge, attnreg
        for i in range(self.W.shape[0]):
            nn.init.xavier_uniform_(self.W[i].data, gain=1.414)

        # for _ in range(nhead):
            # self.W.append(nn.Parameter(torch.zeros(size=(ninfeat, noutfeat))))
            # nn.init.xavier_uniform_(self.W[-1].data, gain=1.414)
            # self.a.append(nn.Linear(2*noutfeat, 1, bias=False))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, normalization_coefficient, verbose_flag, adj):
        '''
        :param x:       FloatTensor B*F*E1
        :param normalization_coefficient: normalization coefficient for regularization
        :verbose_flag: whether to print
        :param adj:     FloatTensor F*F
        :return:        FloatTensor B*F*E2
        '''
        nfield = x.size(1)
        zero_vec = -9e15 * torch.ones_like(adj)
        mask = torch.where(adj > 0, adj, zero_vec)                               # B*F*F, all ones, changed, including batches B in the first dimension, attnreg
        mask = 1/nfield * mask                                                   # this one is intialized to 1/nfield, attnreg, actually equals to attn for GAT, used for summing neighbours!!

        vi_indices, vj_indices = get_all_indices(nfield)
        # h = torch.einsum('bfi,io->bfo', x, self.W[head])                        # B*F*E2
        # vi, vj = h[:, vi_indices], h[:, vj_indices]
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
        
        return torch.cat(h_prime_list, dim=1)                                                          # B*F*(E2)
        # return h_prime                                                           # B*F*E2

# Model:  Attentive No Regularization Model
class AttentiveNoRegModel(nn.Module):
    def __init__(self, nfield, nfeat, nemb, gat_layers, gat_hid, mlp_layers, mlp_hid, dropout, alpha=0.2, nhead=8):
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)

        self.gat_layers = gat_layers
        self.gats = torch.nn.ModuleList()
        ninfeat = nemb
        for _ in range(gat_layers):
            self.gats.append(AttentiveNoRegLayer(nfield, ninfeat, nemb, gat_hid, nhead, dropout, alpha))  # attnreg, pass in nfield
            # ninfeat = nhead*gat_hid  # attnreg no attention heads
            ninfeat = gat_hid

        self.dropout = nn.Dropout(p=dropout)
        self.affine = MLP(nfield*ninfeat, mlp_layers, mlp_hid, dropout)

    def forward(self, x, normalization_coefficient, verbose_flag, adj=None):
        """
        :param x:   {'ids': LongTensor B*F, 'vals': FloatTensor B*F}
        :param normalization_coefficient: normalization coefficient for regularization
        :verbose_flag: whether to print
        :param adj:     FloatTensor F*F, default fully connected
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        h = self.embedding(x)                                                   # B*F*E
        if verbose_flag:
            print ("entering model x norm: ", torch.norm(h))
            print ("entering model torch.norm(self.gats[0].W[0]) norm: ", torch.norm(self.gats[0].W[0]))
        if adj is None:
            adj = torch.ones((h.size(0), h.size(1), h.size(1)), dtype=h.dtype, device=h.device)  # B*F*F attnreg
        for l in range(self.gat_layers):
            h = self.gats[l](h, normalization_coefficient, verbose_flag, adj)                                         # B*F*(gat_hid)
            # print ("h shape: ", h.shape)
            if verbose_flag:
                print ("after aggregating neighbouring nodes h norm: ", torch.norm(h))
            h = F.elu(self.dropout(h))
            if verbose_flag:
                print ("after elu and dropout h norm: ", torch.norm(h))

        y = self.affine(h.view(h.size(0), -1))                                  # B*1
        return y.squeeze(1)                                           # B
