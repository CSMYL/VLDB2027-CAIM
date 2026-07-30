import torch
from torch import nn
import torch.nn.functional as F
from models.layers import get_all_indices, Embedding, MLP
from scipy.special import softmax
from sklearn.utils.extmath import softmax as softmaxsklearn

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

def softmax_func(x, dim):
    maxes = torch.max(x, dim, keepdim=True)[0]  # returns (data, indices)
    print ("maxes shape: ", maxes.shape)
    print ("maxes: ", maxes)
    x_exp = torch.exp(x-maxes)
    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    output_custom = x_exp/x_exp_sum
    return output_custom

def softmax_func_unstab(x, dim):
    x_exp = torch.exp(x)
    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    output_custom = x_exp/x_exp_sum
    return output_custom


if __name__ == '__main__':

    # h_list.append(torch.einsum('bfi,io->bfo', xj[:, i].view(xj.shape[0], 1, -1), self.W[i]))
    x = torch.tensor([[[1., 0.1], [1., 0.1]], [[1., 0.1], [1., 0.1]]], requires_grad=True)
    y = 2 * torch.tensor([[[1., 0.1], [1., 0.1]], [[1., 0.1], [1., 0.1]]], requires_grad=True)
    w = 3.0
    wx = w * x
    w_mul_x = torch.mul(x, w)
    # h = torch.einsum('bfi,io->bfo', x, w)
    print ("x.shape: ", x.shape)
    print ("x: ", x)
    print ("y.shape: ", y.shape)
    print ("y: ", y)
    print ("w: ", w)
    print ("wx: ", wx)
    print ("w_mul_x", w_mul_x)
    wdivx = w/x
    w_div_x = torch.div(w,x)
    print ("wdivx: ", wdivx)
    print ("w_div_x: ", w_div_x)

    h = torch.einsum('xy,xy->xy', x[0], y[0])
    print ("x[0]: \n", x[0])
    print ("y[0]: \n", y[0])
    print ("h shape: \n", h.shape)
    print ("h: \n", h)
    
    x_0 = 10 * x[0]
    z = torch.tensor([[[1., 0.1], [1., 0.1]], [[1., 0.1], [1., 0.1]], [[2., 0.2], [4., 0.4]]], requires_grad=True)
    print ("z: \n", z)
    print ("x_0: \n", x_0)
    print ("(z+x_0): \n", (z+x_0))

    # https://pytorch.org/docs/stable/generated/torch.broadcast_tensors.html
    print ("broadcasting")
    z = torch.randn(2,3)
    w = torch.randn(3,1,1)
    print ("z.shape: \n", z.shape)
    print ("z: \n", z)
    print ("w.shape: \n", w.shape)
    print ("w: \n", w)
    c, d = torch.broadcast_tensors(z, w)
    print ("c shape: \n", c.shape)
    print ("c: \n", c)
    print ("d shape: \n", d.shape)
    print ("d: \n", d)

    print ("broadcasting again")
    z = torch.randn(3,2,2)
    w = torch.ones(2,2)
    print ("z: ", z)
    print ("w: ", w)
    z_w_sum = z + w
    print ("z_w_sum: ", z_w_sum)
    
    print ("softmax functions")
    x = torch.randn(2, 3, 4)
    print ("x: ", x)
    output_torch = F.softmax(x, -1)
    output_func = softmax_func(x, -1)
    output_func_unstab = softmax_func_unstab(x, -1)
    output_scipy = softmax(x.numpy(), axis=-1)
    
    
    print ("output_torch: ", output_torch)
    print ("output_func: ", output_func)
    print ("output_func_unstab: ", output_func_unstab)
    print ("output_scipy: ", output_scipy)
    # print ("softmax sum 0: ", torch.sum(output2, dim=0))
    # print ("softmax sum 1: ", torch.sum(output2, dim=1))
    # print ("softmax sum -1 shape: ", torch.sum(output2, dim=-1, keepdim=True).shape)
    # print ("softmax sum -1: ", torch.sum(output2, dim=-1, keepdim=True))
    # print ("softmax sum -2 shape: ", torch.sum(output2, dim=-2, keepdim=True).shape)
    # print ("softmax sum -2: ", torch.sum(output2, dim=-2, keepdim=True))

    x = torch.randn(3, 4)
    print ("x: ", x)
    output_sklearn_1 = softmaxsklearn(x.numpy())
    output_func_1 = softmax_func(x, 1)
    print ("output_sklearn_1: ", output_sklearn_1)
    print ("output_func_1: ", output_func_1)

    #### norm(2)
    z = torch.tensor([[[1., 2.], [3., 4.]], [[-1., -2.], [-3., -4.]] ], requires_grad=True)
    print ("z: \n", z)
 
    z_norm = z.norm(2) * z.norm(2)
    z_pow = torch.pow(z, 2).sum()
    print ("z_norm only: ", z.norm(2))
    print ("z_norm: ", z_norm)
    print ("z pow only: ", torch.pow(z, 2))
    print ("z_pow: ", z_pow)

    #### exp
    z = torch.tensor([[[1., 2.], [3., 4.]], [[-1., -2.], [-3., -4.]] ], requires_grad=True)
    print ("z: \n", z)
 
    z_exp = torch.exp(z)
    print ("z_exp shape: ", z_exp.shape)
    print ("z_exp: ", z_exp)

    #### log
    z = torch.tensor([[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]] ], requires_grad=True)
    print ("z: \n", z)
 
    z_log = torch.log(z)
    print ("z_log shape: ", z_log.shape)
    print ("z_log: ", z_log)
    print ("torch.exp(1.): ", torch.exp(torch.tensor([[[1., 1.]]])))
    print ("torch.log(torch.exp(1.)): ", torch.log(torch.exp(torch.tensor([[[1., 1.]]]))))

    #### torch norm
    xi_prime = torch.tensor([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]], requires_grad=True)
    print ("xi_prime shape: ", xi_prime.shape)
    print ("xi_prime: \n", xi_prime)
    xi_prime_norm = xi_prime / torch.norm(xi_prime, dim=-1, keepdim=True)
    print ("torch.norm(xi_prime, dim=-1, keepdim=True) shape: ", torch.norm(xi_prime, dim=-1, keepdim=True).shape)
    print ("xi_prime_norm shape: ", xi_prime_norm.shape)
    print ("xi_prime_norm: \n", xi_prime_norm)

    vi_indices = [0,1,0,1]
    xi_prime_transform = xi_prime[:, vi_indices]
    print ("xi_prime_transform shape: ", xi_prime_transform.shape)
    print ("xi_prime_transform: \n", xi_prime_transform)
    xi_prime_transform_norm = xi_prime_transform / torch.norm(xi_prime_transform, dim=-1, keepdim=True)
    print ("torch.norm(xi_prime_transform, dim=-1, keepdim=True) shape: ", torch.norm(xi_prime_transform, dim=-1, keepdim=True).shape)
    print ("torch.norm(xi_prime_transform, dim=-1, keepdim=True): \n", torch.norm(xi_prime_transform, dim=-1, keepdim=True))
    print ("xi_prime_transform_norm shape: ", xi_prime_transform_norm.shape)
    print ("xi_prime_transform_norm: \n", xi_prime_transform_norm)

"""
from scipy.special import softmax
x = np.array([[1,2,2], [7,8,8]])
import numpy as np
x = np.array([[1,2,2], [7,8,8]])
softmax(x, axis=1)
array([[0.1553624, 0.4223188, 0.4223188],
       [0.1553624, 0.4223188, 0.4223188]])
"""
