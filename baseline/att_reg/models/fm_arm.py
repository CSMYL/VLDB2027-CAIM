import torch
import torch.nn as nn
from utils.entmax import EntmaxBisect
from models.layers import Embedding, FactorizationMachine, MLP

class SparseAttLayer(nn.Module):
    def __init__(self, nfield, nemb, nhid, alpha=1.5):
        """ Sparse Attention Layer """
        super(SparseAttLayer, self).__init__()
        if alpha == 1.:
            self.sparsemax = nn.Softmax(dim=-1)
        else:
            self.sparsemax = EntmaxBisect(alpha, dim=-1)

        self.Q = nn.Parameter(torch.zeros(nhid, nemb))              # O*E
        nn.init.xavier_uniform_(self.Q, gain=1.414)

        self.bilinear = nn.Parameter(torch.zeros(nemb, nemb))       # E*E
        nn.init.xavier_uniform_(self.bilinear, gain=1.414)

        self.values = nn.Parameter(torch.zeros(nhid, nfield))       # O*F
        nn.init.xavier_uniform_(self.values, gain=1.414)

    def forward(self, x):
        """
        :param x:  B*F*E
        :return:        Att_weights (B*O*F), Key (B*F*E) <-> Q (O*E) -> W (O*F)
        """
        keys = x                                                    # B*F*E

        # sparse gates
        att_gates = torch.einsum('bfx,xy,oy->bof',
                                 keys, self.bilinear, self.Q)       # B*O*F
        sparse_gates = self.sparsemax(att_gates)                    # B*O*F

        return torch.einsum('bof,of->bof', sparse_gates, self.values)

class FM_ARMModel(nn.Module):
    """
    Model:  Adaptive Relation Modeling Network
    """
    def __init__(self, nfield, nfeat, nemb, nhead, alpha, arm_hid, mlp_layers, mlp_hid,
                 dropout, ensemble, deep_layers, deep_hid):
        super().__init__()
        self.nfield, self.nfeat, self.nemb = nfield, nfeat, nemb
        self.nhead, self.arm_hid = nhead, arm_hid
        self.ensemble = ensemble
        self.dropout = nn.Dropout(p=dropout)

        # fm
        self.fm_emb = Embedding(nfeat, nemb)
        self.fm = FactorizationMachine(reduce_dim=False)

        # embedding
        self.embedding = Embedding(nfeat, nemb)
        self.emb_bn = nn.BatchNorm1d(nfield)

        # arm
        self.attn_layers = nn.ModuleList([
            SparseAttLayer(nfield, nemb, arm_hid, alpha) for _ in range(nhead)
        ])
        self.arm_bn = nn.BatchNorm1d(nhead*arm_hid)

        # MLP
        self.mlp = MLP(nhead*arm_hid*nemb+nemb, mlp_layers, mlp_hid, dropout)

        if ensemble:
            self.deep_embedding = Embedding(nfeat, nemb)
            self.deep_mlp = MLP(nfield*nemb, deep_layers, deep_hid, dropout)
            self.ensemble_layer = nn.Linear(2, 1)
            nn.init.constant_(self.ensemble_layer.weight, 0.5)
            nn.init.constant_(self.ensemble_layer.bias, 0.)

    def forward(self, x):
        """
        :param x:   {'ids': LongTensor B*F, 'vals': FloatTensor B*F}
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        x['vals'].clamp_(0.001, 1.)
        x_emb = self.embedding(x)                                       # B*F*E

        x_exp = self.emb_bn(torch.exp(x_emb))                           # B*F*E
        arm_weights = [layer(x_emb) for layer in self.attn_layers]      # K (B*O*F)
        arm_weights = torch.stack(arm_weights, dim=1)                   # B*K*O*F

        arm = torch.einsum('bfe,bkof->bkoe', x_exp, arm_weights)        # B*K*O*E
        arm = arm.view(-1, self.nhead*self.arm_hid, self.nemb)          # B*(KxO)*E
        arm = self.arm_bn(arm).view(arm.size(0), -1)                    # B*(KxOxE)
        arm = self.dropout(arm)

        fm = self.fm(self.fm_emb(x))                                    # B*E
        # concatenate fm and arm
        y = self.mlp(torch.cat([arm, fm], dim=1))                       # B*1

        if self.ensemble:
            deep_emb = self.deep_embedding(x)
            y_deep = self.deep_mlp(
                deep_emb.view(-1, self.nfield*self.nemb))               # B*1

            y = torch.cat([y, y_deep], dim=1)                           # B*2
            y = self.ensemble_layer(y)                                  # B*1

        y = y.squeeze(1)                                                # B
        return y