import torch
from torch import nn
import torch.nn.functional as F
from models.layers import get_all_indices, Embedding, MLP
import numpy as np
import time
import datetime

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


def cuda(v):
    if torch.cuda.is_available(): return v.cuda()
    return v

if __name__ == '__main__':

    torch.manual_seed(2020)
    ### example 1
    a = torch.tensor([2., 3.], requires_grad=True)
    b = torch.tensor([6., 4.], requires_grad=True)

    Q = 3*a**3 - b**2

    external_grad = torch.tensor([1., 1.])
    Q.backward(gradient=external_grad)

    print ("9*a**2: ", 9*a**2)
    print ("-2*b: ", -2*b)

    print ("9*a**2==a.grad: ", 9*a**2==a.grad)
    print ("-2*b==b.grad: ", -2*b==b.grad)

    ### example 2
    x = torch.rand(5, 5)
    y = torch.rand(5, 5)
    z = torch.rand((5, 5), requires_grad=True)

    a = x + y
    print(f"Does `a` require gradients? : {a.requires_grad}")
    print("a: ", a)
    b = x + z
    print(f"Does `b` require gradients?: {b.requires_grad}")
    print("b: ", b)


    ### example 3
    x = torch.ones(2, 2, requires_grad=True)
    print(x)

    print(x.data)

    print(x.grad)

    print(x.grad_fn)  # we've created x ourselves


    y = x + 2
    print(y)

    print(y.grad_fn)

    z = y * y * 3
    out = z.mean()

    print(z, out)

    a = torch.randn(2, 2)
    a = ((a * 3) / (a - 1))
    print(a.requires_grad)
    a.requires_grad_(True)
    print(a.requires_grad)
    b = (a * a).sum()
    print(b.grad_fn)

    out.backward()
    print("not cat x.grad: ", x.grad)


    x = torch.ones(2, 2, requires_grad=True)
    y = x + 2
    y.backward(torch.ones(2, 2), retain_graph=True)
    # y.backward(torch.ones(2, 2))
    # the retain_variables flag will prevent the internal buffers from being freed
    # print("not retain: ", x.grad)
    print("retain_graph: ", x.grad)

    z = y * y
    print(z)


    gradient = torch.randn(2, 2)

    # this would fail if we didn't specify
    # that we want to retain variables
    y.backward(gradient)

    print(x.grad)


    print(x.requires_grad)
    print((x ** 2).requires_grad)

    with torch.no_grad():
        print((x ** 2).requires_grad)

    ### example 4

    x = torch.ones(5)  # input tensor
    y = torch.zeros(3)  # expected output
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    z = torch.matmul(x, w)+b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

    print('Gradient function for z =',z.grad_fn)
    print('Gradient function for loss =', loss.grad_fn)


    loss.backward()
    print(w.grad)
    print(b.grad)

    z = torch.matmul(x, w)+b
    print(z.requires_grad)

    with torch.no_grad():
        z = torch.matmul(x, w)+b
        print (z.requires_grad)
    print(z.requires_grad)

    z = torch.matmul(x, w)+b
    print ("z grad: ", z.requires_grad)
    z_det = z.detach()
    print(z_det.requires_grad)

    print ("z grad after detach: ", z.requires_grad)

    ### only affect the results of computation
    with torch.no_grad():
        print ("no grad z: ", z.requires_grad)
    print ("after no grad z: ", z.requires_grad)

    ### concat
    print ("concatenate")
    x1 = torch.ones(1, 2, requires_grad=True)
    x2 = torch.zeros(1, 2, requires_grad=True)
    print(x1)
    print(x2)

    print(x1.data)
    print(x2.data)

    print(x1.grad)
    print(x2.grad)

    print(x1.grad_fn)  # we've created x ourselves
    print(x2.grad_fn)


    y1 = x1 + 2
    print(y1)
    y2 = x2 + 2
    print(y2)

    print(y1.grad_fn)
    print(y2.grad_fn)

    y = torch.cat([y1, y2], dim=0)
    print ("y: ", y)
    z = y * y * 3

    print ("z: ", z)
    z = z.view(-1, 4)
    print ("view z: ", z)
    z = z.view(2,2)
    print ("view again z: ", z)
    out = z.mean()

    print(z, out)

    out.backward()
    print("cat x1.grad: ", x1.grad)
    print("cat x2.grad: ", x2.grad)

    ### Softmax
    x = torch.ones(2, 5, requires_grad=True)
    y = F.softmax(x, dim=-1)
    print ("y: ", y)
    y.backward(torch.ones(2,5))
    print ("x.grad: ", x.grad)

    ### inverse
    print ("inverse")
    x = torch.tensor([[2.,2.],[2., 2.]], requires_grad=True)
    y = 1./x
    print ("y: ", y)
    y.backward(torch.ones(2,2))
    print ("x.grad: ", x.grad)

    ### several losses added
    print ("sevearl losses together")
    x = torch.ones(2, 2, requires_grad=True)
    print(x)
    print(x.data)
    print(x.grad)
    print(x.grad_fn)  # we've created x ourselves
    y = x + 2
    print(y)
    print(y.grad_fn)
    sum_loss = None
    print ("sum_loss: ", sum_loss)
    z1 = y * y * 3
    out1 = z1.mean()
    print(z1, out1)
    sum_loss = out1
    print ("after out1 sum loss: ", sum_loss)
    z2 = y * y * 4
    out2 = z2.mean()
    print(z2, out2)
    sum_loss = sum_loss + out2
    print ("after out2 sum loss: ", sum_loss)
    z3 = y * y * 5
    out3 = z3.mean()
    print(z3, out3)
    sum_loss = sum_loss + out3
    print ("after out3 sum loss: ", sum_loss)

    sum_loss.backward()
    print("x.grad: ", x.grad)

    ### several losses separately
    print ("sevearl losses separately")

    ###
    w = torch.ones(2, 2, requires_grad=True)
    x = torch.ones(2, 2, requires_grad=True)
    k = torch.matmul(w, x)
    ###
    print(x)
    print(x.data)
    print(x.grad)
    print(x.grad_fn)  # we've created x ourselves
    y = x + 2
    print(y)
    print(y.grad_fn)
    sum_loss = None
    print ("sum_loss: ", sum_loss)
    z1 = y * y * 3
    out1 = z1.mean()
    print(z1, out1)
    out1.backward()
    print ("after out1 grad: ", x.grad)

    z2 = y * y * 4
    out2 = z2.mean()
    print(z2, out2)
    # sum_loss = sum_loss + out2
    out2.backward()
    print ("after out2 grad: ", x.grad)
    
    z3 = y * y * 5
    out3 = z3.mean()
    print(z3, out3)
    # sum_loss = sum_loss + out3
    out3.backward()
    print ("after out3 grad: ", x.grad)

    # sum_loss.backward()
    # print("x.grad: ", x.grad)

    
    ### norm(2)
    print ("norm(2)")
    x = torch.tensor([[2.,2.],[2., 2.]], requires_grad=True)
    reg_strength = 3
    y = reg_strength * x.norm(2) * x.norm(2)
    norm_val = x.norm(2)
    print ("norm_val: ", norm_val)
    print ("y: ", y)
    y.backward()
    print ("x.grad: ", x.grad)


    
    ### bfi,fio->bfo, gradient backward comparing einsum and "for"
    batch_size = 1024
    field = 40 * 40
    input_dim = 25
    output_dim =32
    # h_in_np = np.random.rand(batch_size, field, input_dim)
    # weights_np = np.random.rand(field, input_dim, output_dim)
    h_in_np = np.ones((batch_size, field, input_dim))
    weights_np = np.ones((field, input_dim, output_dim))
    h_in_np_for = np.ones((batch_size, field, input_dim))
    weights_np_for = np.ones((field, input_dim, output_dim))
    
    ### einsum gpu
    print ("einsum gpu multiply")
    h_in_ein = cuda(torch.tensor(h_in_np, requires_grad=True))
    weights_ein = cuda(torch.tensor(weights_np, requires_grad=True))
    start = time.time()
    st_str = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')     
    print ("before forward time: ", st_str)
    h_out_ein = torch.einsum('bfi,fio->bfo', h_in_ein, weights_ein)
    print ("h_out_ein shape: ", h_out_ein.shape)
    h_out_ein_mean = h_out_ein.mean()
    end = time.time()
    end_str = datetime.datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S')
    print ("after forward time: ", end_str)
    print ("forward duration: ", end-start)
    h_out_ein_mean.backward()
    end = time.time()
    end_str = datetime.datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S')
    print ("after backward time: ", end_str)
    print ("forward+backward duration: ", end-start)
    print ()
 
    ### for gpu
    print ("for gpu multiply")
    h_in_for = cuda(torch.tensor(h_in_np_for, requires_grad=True))
    weights_for = cuda(torch.tensor(weights_np_for, requires_grad=True))
    print ("before forward time: ", st_str)
    h_list = []
    for i in range(weights_for.shape[0]):
        h_list.append(torch.einsum('bfi,io->bfo', h_in_for[:, i].view(h_in_for.shape[0], 1, -1), weights_for[i]))
    h_out_for = torch.cat(h_list, dim=1)
    print ("h_out_for shape: ", h_out_for.shape)
    h_out_for_mean = h_out_for.mean()
    end = time.time()
    end_str = datetime.datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S')
    print ("after forward time: ", end_str)
    print ("forward duration: ", end-start)
    h_out_for_mean.backward()
    end = time.time()
    end_str = datetime.datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S')
    print ("after backward time: ", end_str)
    print ("forward+backward duration: ", end-start)

    ### bfi,fio->bfo, gradient backward comparing einsum and "for"
    batch_size = 1024
    field = 40 * 40
    input_dim = 25
    output_dim =32
    # h_in_np = np.random.rand(batch_size, field, input_dim)
    # weights_np = np.random.rand(field, input_dim, output_dim)
    h_in_np = np.ones((batch_size, field, input_dim))
    weights_np = np.ones((field, input_dim, output_dim))
    h_in_np_for = np.ones((batch_size, field, input_dim))
    weights_np_for = np.ones((field, input_dim, output_dim))
    print ()

    ### einsum cpu
    print ("einsum cpu multiply")
    h_in_ein = torch.tensor(h_in_np, requires_grad=True)
    weights_ein = torch.tensor(weights_np, requires_grad=True)
    start = time.time()
    st_str = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')     
    print ("before forward time: ", st_str)
    h_out_ein = torch.einsum('bfi,fio->bfo', h_in_ein, weights_ein)
    print ("h_out_ein shape: ", h_out_ein.shape)
    h_out_ein_mean = h_out_ein.mean()
    end = time.time()
    end_str = datetime.datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S')
    print ("after forward time: ", end_str)
    print ("forward duration: ", end-start)
    h_out_ein_mean.backward()
    end = time.time()
    end_str = datetime.datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S')
    print ("after backward time: ", end_str)
    print ("forward+backward duration: ", end-start)
    print ()
 
    ### for cpu
    print ("for cpu multiply")
    h_in_for = torch.tensor(h_in_np_for, requires_grad=True)
    weights_for = torch.tensor(weights_np_for, requires_grad=True)
    print ("before forward time: ", st_str)
    h_list = []
    for i in range(weights_for.shape[0]):
        h_list.append(torch.einsum('bfi,io->bfo', h_in_for[:, i].view(h_in_for.shape[0], 1, -1), weights_for[i]))
    h_out_for = torch.cat(h_list, dim=1)
    print ("h_out_for shape: ", h_out_for.shape)
    h_out_for_mean = h_out_for.mean()
    end = time.time()
    end_str = datetime.datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S')
    print ("after forward time: ", end_str)
    print ("forward duration: ", end-start)
    h_out_for_mean.backward()
    end = time.time()
    end_str = datetime.datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S')
    print ("after backward time: ", end_str)
    print ("forward+backward duration: ", end-start)

"""
from scipy.special import softmax
x = np.array([[1,2,2], [7,8,8]])
import numpy as np
x = np.array([[1,2,2], [7,8,8]])
softmax(x, axis=1)
array([[0.1553624, 0.4223188, 0.4223188],
       [0.1553624, 0.4223188, 0.4223188]])
"""
