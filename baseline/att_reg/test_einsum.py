import torch
from torch import nn
import torch.nn.functional as F
from models.layers import get_all_indices, Embedding, MLP
import numpy as np

class AttentiveRegLayer(nn.Module):

    def __init__(self, nfield, ninfeat, nemb, noutfeat, nhead, dropout, alpha):
        super().__init__()
        
        self.W_Q = nn.Parameter(torch.zeros(ninfeat, nemb))       # E1*E, only this layer, attnreg
        nn.init.xavier_uniform_(self.W_Q, gain=1.414)

        self.W_Key = nn.Parameter(torch.zeros(ninfeat, nemb))       # E1*E, only this layer, attnreg
        nn.init.xavier_uniform_(self.W_Key, gain=1.414)

        self.values = nn.Parameter(torch.zeros(nfield, nfield))       # F*F, including self-edge, attnreg
        nn.init.xavier_uniform_(self.values, gain=1.414)

        self.W = nn.Parameter(torch.zeros(size=(nfield*nfield, ninfeat, noutfeat)))  # (FXF)*E1*E2, assume all connected ..., including self-edge, attnreg
        for i in range(self.W.shape[0]):
            nn.init.xavier_uniform_(self.W[i].data, gain=1.414)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj):
        '''
        :param x:       FloatTensor B*F*E1
        :param adj:     FloatTensor F*F
        :return:        FloatTensor B*F*E2
        '''
        nfield = x.size(1)
        zero_vec = -9e15 * torch.ones_like(adj)
        mask = torch.where(adj > 0, adj, zero_vec)                               # B*F*F, all ones, changed, including batches B in the first dimension, attnreg
        mask = 1/nfield * mask                                                   # this one is intialized to 1/nfield, attnreg, actually equals to attn for GAT, used for summing neighbours!!
        # print ("mask???: ", mask)  # attnreg

        vi_indices, vj_indices = get_all_indices(nfield)
        xi, xj = x[:, vi_indices], x[:, vj_indices]                               # xj: B*(FxF)*(E1): [0,1,2,3,4,5..,9,0,1,2,3,4,5..,9,0,1,2,3,4,5..,9,0,1,2,3,4,5..,9]
        ### x * w --> for neighbouring nodes
        h_list = []
        for i in range(self.W.shape[0]):                                          # FxF
            h_list.append(torch.einsum('bfi,io->bfo', xj[:, i].view(xj.shape[0], 1, -1), self.W[i]))        # B*(1)*E2, attnreg 1: x[:, i] shape ??, attnreg 2: torch.cat(customized_h_list, dim=1) ??
        h = torch.cat(h_list, dim=1)                                              # B*(FxF)*E2

        ### using mask for aggregating for central nodes
        h_prime_list = []
        for i in range(mask.shape[1]):
            h_prime = torch.einsum('bxy,bye->bxe', mask[:,i].view(mask.shape[0],1,-1), h[:,i*nfield:(i+1)*nfield])                           # B*1*E2
            h_prime_list.append(h_prime)
        
        ### attentive regularization
        query = torch.einsum('bfi,io->bfo', xi, self.W_Q)                         # B*(FxF)*E2
        key = torch.einsum('bfi,io->bfo', xj, self.W_Key)                         # B*(FxF)*E2
        attn = torch.einsum('bfo,bfo->bf', query, key)                            # B*(FxF)*1
        attn = self.dropout(F.softmax(attn.view(-1,nfield,nfield), dim=-1))       # B*F*F
        attn = 1.0 / attn                                                         # B*F*F --> inverse
        reg_strength = torch.einsum('bxy,xy->bxy', attn, self.values)             # B*F*F
        reg_strength = reg_strength.view(reg_strength.shape[0], -1)               # B*(FxF)
        reg_strength = torch.mean(reg_strength, dim=0)                            # (FxF) average over all samples in this batch ...
        layer_reg_loss = None
        for i in range(reg_strength.shape[0]):
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

    def forward(self, x, adj=None):
        """
        :param x:   {'ids': LongTensor B*F, 'vals': FloatTensor B*F}
        :param adj:     FloatTensor F*F, default fully connected
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        reg_values = None
        h = self.embedding(x)                                                   # B*F*E
        if adj is None:
            adj = torch.ones((h.size(0), h.size(1), h.size(1)), dtype=h.dtype, device=h.device)  # B*F*F attnreg
        for l in range(self.gat_layers):
            h, layer_reg_values = self.gats[l](h, adj)                                         # B*F*(gat_hid)
            # print ("h shape: ", h.shape)
            h = F.elu(self.dropout(h))
            if reg_values is None:
                reg_values = layer_reg_values
            else:
                reg_values = reg_values + layer_reg_values

        y = self.affine(h.view(h.size(0), -1))                                  # B*1
        return y.squeeze(1), reg_values                                         # B

def get_triu_indices(n, diag_offset=1):
    """get the row, col indices for the upper-triangle of an (n, n) array"""
    return np.triu_indices(n, diag_offset)


if __name__ == '__main__':

    # h_list.append(torch.einsum('bfi,io->bfo', xj[:, i].view(xj.shape[0], 1, -1), self.W[i]))
    x = torch.tensor([[[1., 0.1], [1., 0.1]], [[1., 0.1], [1., 0.1]]])
    w = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    h = torch.einsum('bfi,io->bfo', x, w)
    print ("x.shape: ", x.shape)
    print ("w.shape: ", w.shape)
    print ("bfi,io->bfo")
    print ("h: \n", h)

    # attn = torch.einsum('bfo,bfo->bf', query, key)
    query = torch.tensor([[[1., 0.1], [2., 0.2]], [[1., 0.1], [2., 0.2]]])
    key = torch.tensor([[[3., 0.3], [4., 0.4]], [[3., 0.3], [4., 0.4]]])
    attn = torch.einsum('bfo,bfo->bf', query, key)
    print ("query.shape: ", query.shape)
    print ("key.shape: ", key.shape)
    print ("bfo,bfo->bf")
    print ("attn: \n", attn)

    # attn.view(-1,nfield,nfield)
    attn = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    print ("attn.view(-1,2,2): \n", attn.view(-1,2,2))

    # (F.softmax(attn.view(-1,nfield,nfield), dim=-1))
    attn = torch.tensor([[1., 2., 2., 4., 4., 4.], [7., 8., 8., 10., 10., 10.]])
    print ("F.softmax(attn.view(-1,2,3), dim=-1): \n", F.softmax(attn.view(-1,2,3), dim=-1))

    # reg_strength = torch.einsum('bxy,xy->bxy', attn, self.values)
    attn = torch.tensor([[[1., 0.1], [2., 0.2]], [[3., 0.3], [4., 0.4]]])
    values = torch.tensor([[1., 0.1], [2., 0.2]])
    reg_strength = torch.einsum('bxy,xy->bxy', attn, values)
    print ("attn: ", attn)
    print ("values: ", values)
    print ("bxy,xy->bxy")
    print ("reg_strength: \n", reg_strength)
    

    reg_strength = torch.tensor([[1., 0.1], [2., 0.2], [3., 0.3]])
    print ("reg_strength.shape: ", reg_strength.shape)
    print ("reg_strength: \n", reg_strength)
    reg_strength = torch.mean(reg_strength, dim=0)
    print ("reg_strength shape: ", reg_strength.shape)
    print ("reg_strength: \n", reg_strength)
    

    mask = torch.tensor([[[0.3, 0.3, 0.4]], [[0.2, 0.2, 0.6]], [[1., 1., 1.]]])
    h = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]], [[7., 8.], [9.,10.], [11., 12.]], [[2., 2.], [2., 2.], [2., 2.]]])
    print ("mask shape: \n", mask.shape)
    print ("h shape: \n", h.shape)
    # h_prime = torch.einsum('bxy,bye->bxe', mask[:,i].view(mask.shape[0],1,-1), h[:,i*nfield:(i+1)*nfield])
    h_prime = torch.einsum('bxy,bye->bxe', mask, h)
    print ("bxy,bye->bxe")
    print ("h_prime shape: \n", h_prime.shape)
    print ("h_prime: \n", h_prime)


    h_in = torch.tensor([[[1., 0.1], [2., 0.2], [3., 0.3]], [[4., 0.4], [5., 0.5], [6., 0.6]]])
    weights = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]])
    print ("h_in shape: \n", h_in.shape)
    print ("weights shape: \n", weights.shape)
    # h_prime = torch.einsum('bxy,bye->bxe', mask[:,i].view(mask.shape[0],1,-1), h[:,i*nfield:(i+1)*nfield])
    h_out = torch.einsum('bfi,fio->bfo', h_in, weights)
    print ("bfi,fio->bfo")
    print ("h_out shape: \n", h_out.shape)
    print ("h_out: \n", h_out)
    
    h_list = []
    for i in range(weights.shape[0]):
        h_list.append(torch.einsum('bfi,io->bfo', h_in[:, i].view(h_in.shape[0], 1, -1), weights[i]))
    h_out = torch.cat(h_list, dim=1)
    print ("using for")
    print ("h_out shape: \n", h_out.shape)
    print ("h_out: \n", h_out)
    
    ### many operands (n-th power): https://stackoverflow.com/questions/55894693/understanding-pytorch-einsum
    aten = torch.tensor([[2., 2.], [3., 3.]])
    aten_power = torch.einsum('ij, ij, ij, ij -> ij', aten, aten, aten, aten)
    print ("aten: \n", aten)
    print ("aten_power shape: \n", aten_power.shape)
    print ("aten_power: \n", aten_power)
    
    ### step 1: neighbours multiplying weights
    
    ### einsum
    x = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]) # B * F_j * E1
    W = torch.tensor([[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], [[[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]]]]) # F_i * F_j * E1 * E2
    h = torch.einsum('bjt,ijtk->bijk', x, W)  # B * F_i * F_j * E2
    print ("einsum \n")
    print ("before reshape h: \n", h)
    h = torch.reshape(h, (2,4,2))  # B * (F_i * F_j) * E2
    print ("h shape: \n", h.shape)
    print ("h: \n", h)   

    h_reshape = torch.einsum('bjt,ijtk->bijk', x, torch.reshape(W, (2, 2, 2, -1)))  # B * F_i * F_j * E2
    print ("einsum \n")
    print ("before reshape h_reshape: \n", h_reshape)
    h_reshape = torch.reshape(h_reshape, (2,4,2))  # B * (F_i * F_j) * E2
    print ("h_reshape shape: \n", h_reshape.shape)
    print ("h_reshape: \n", h_reshape)

    ### original code
    nfield = 2
    vi_indices, vj_indices = get_all_indices(nfield) 
    xj = x[:, vj_indices]
    W_ori = torch.reshape(W, (4,2,2))
    h_list = []
    for i in range(W_ori.shape[0]):
        h_list.append(torch.einsum('bfi,io->bfo', xj[:, i].view(xj.shape[0], 1, -1), W_ori[i]))
    h = torch.cat(h_list, dim=1)  # B * (F_i * F_j) * E2
    print ("original code \n")
    print ("h shape: \n", h.shape)
    print ("h: \n", h)
    
    ### step 2: using mask for aggregating neighbouring nodes for central nodes
    ### directly average
    h_avg = torch.mean(h.view(2,2,2,2), dim=-2)  # B * (F_i * F_j) * E2 --> B * F_i * E2
    print ("average\n")
    print ("h_avg shape: \n", h_avg.shape)
    print ("h_avg: \n", h_avg)

    h_avg_reshape = torch.mean(torch.reshape(h, (2,2,2,2)), dim=-2)  # B * (F_i * F_j) * E2 --> B * F_i * E2
    print ("average\n")
    print ("h_avg_reshape shape: \n", h_avg_reshape.shape)
    print ("h_avg_reshape: \n", h_avg_reshape)

    ### original code
    adj = torch.ones((2, nfield, nfield))  # B*F*F attnreg
    zero_vec = -9e15 * torch.ones_like(adj)
    mask = torch.where(adj > 0, adj, zero_vec)
    mask = 1/nfield * mask
    h_prime_list = []
    for i in range(mask.shape[1]):
        h_prime = torch.einsum('bxy,bye->bxe', mask[:,i].view(mask.shape[0],1,-1), h[:,i*nfield:(i+1)*nfield])
        h_prime_list.append(h_prime)
    h_prime_cat=torch.cat(h_prime_list, dim=1)  # B * F_i * E2
    print ("original code \n")
    print ("h_prime_cat shape \n", h_prime_cat.shape)
    print ("h_prime_cat: \n", h_prime_cat)

    ### step 3 and 4 --> calculate cosine
    ### einsum
    attn = torch.einsum('bik,bijk->bij', h_prime_cat, h.view(2,2,2,2))  # B * F * F
    print ("\n eisum \n")
    print ("attn shape: \n", attn.shape)
    print ("attn: \n", attn)

    ### original code --> using expanding
    xi_prime = h_prime_cat[:, vi_indices]  # B*(FxF)*E2, [0,0,..,1,1,...,2,2,..,9,9..]
    xj_prime = h  # B * (F_i * F_j) * E2
    attn = torch.einsum('bfo,bfo->bf', xi_prime, xj_prime)  # B * (F_i * F_j)
    attn = attn.view(2,2,2)  # B * F * F
    print ("\n original code \n")
    print ("attn shape: \n", attn.shape)
    print ("attn: \n", attn)

    ### step 5 regularized weight norm
    ### einsum
    reg_strength = torch.arange(4) + 1.0
    print ("reg_strength: \n", reg_strength)
    layer_reg_loss = torch.einsum('f,f->', reg_strength, torch.square(torch.norm(W_ori, dim=(1,2))))  # (F*F)*E1*E2 --> (F*F)
    print ("einsum using norm\n")
    print ("layer_reg_loss: \n", layer_reg_loss)
    layer_reg_loss = torch.einsum('f,f->', reg_strength, torch.sum(torch.square(W_ori), dim=(1,2)))  # (F*F)*E1*E2 --> (F*F)
    print ("einsum using sum\n")
    print ("layer_reg_loss: \n", layer_reg_loss)
    ### https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
    ### https://androidkt.com/how-to-add-l1-l2-regularization-in-pytorch-loss-function/
    layer_reg_loss = torch.einsum('f,f->', reg_strength, torch.sum(W_ori.pow(2.0), dim=(1,2)))  # (F*F)*E1*E2 --> (F*F)
    print ("einsum using pow\n")
    print ("layer_reg_loss: \n", layer_reg_loss)

    ### original code
    layer_reg_loss = None
    for i in range(reg_strength.shape[0]):
        if layer_reg_loss is None:
            layer_reg_loss = (reg_strength[i] * W_ori[i].norm(2) * W_ori[i].norm(2))
        else:
            layer_reg_loss = layer_reg_loss + (reg_strength[i] * W_ori[i].norm(2) * W_ori[i].norm(2))
    print ("original code \n")
    print ("layer_reg_loss: \n", layer_reg_loss)

    ### suplement step 1 --> attreg cosine
    print ("suplement step 1 --> attreg cosine")
    x = torch.tensor([[[1., 20.], [30., 4.]], [[5., 66.], [77., 8.]]]) # B * F_j * E1
    nfield = 2
    vi_indices, vj_indices = get_all_indices(nfield) 
    xj = x[:, vj_indices]

    # xi_norm = x / torch.norm(x, dim=-1, keepdim=True)  # B*F*E1
    # xj_norm = xj / torch.norm(xj, dim=-1, keepdim=True) # xj: B*(FxF)*(E1)
    xi_norm = x / 1.0
    xj_norm = xj / 1.0

    ### original einsum
    attn = torch.einsum('bik,bijk->bij', xi_norm, xj_norm.view(x.shape[0], nfield, nfield, -1)).view(x.shape[0], -1)  # B * F * F --> B*(F*F)
    print ("\n origin eisum \n")
    print ("attn shape: \n", attn.shape)
    print ("attn: \n", attn)
 
    ### more simlified einsum
    # xj_norm = x / torch.norm(x, dim=-1, keepdim=True) # xj: B*F*E1
    xj_norm = x / 1.0
    attn = torch.einsum('bik,bjk->bij', xi_norm, xj_norm).view(x.shape[0], -1)  # B * F * F --> B*(F*F)
    print ("\n simplfied eisum \n")
    print ("attn shape: \n", attn.shape)
    print ("attn: \n", attn)

"""
from scipy.special import softmax
x = np.array([[1,2,2], [7,8,8]])
import numpy as np
x = np.array([[1,2,2], [7,8,8]])
softmax(x, axis=1)
array([[0.1553624, 0.4223188, 0.4223188],
       [0.1553624, 0.4223188, 0.4223188]])
"""
