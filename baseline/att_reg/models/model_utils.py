import torch
from models.lr import LRModel
from models.fm import FMModel
from models.hofm import HOFMModel
from models.afm import AFMModel
from models.dcn import CrossNetModel
from models.xdfm import CINModel

from models.dnn import DNNModel
from models.gcn import GCNModel
from models.gat import GATModel

from models.wd import WDModel
from models.pnn import IPNNModel
from models.pnn import KPNNModel
from models.nfm import NFMModel
from models.dfm import DeepFMModel
from models.dcn import DCNModel
from models.xdfm import xDeepFMModel

from models.afn import AFNModel
from models.armnet import ARMNetModel
from models.fm_arm import FM_ARMModel
from models.attreg import AttentiveRegModel
from models.attnoreg import AttentiveNoRegModel

def create_model(args, logger):
    logger.info(f'=> creating model {args.model}')
    if args.model == 'lr':
        model = LRModel(args.nfeat)
    elif args.model == 'fm':
        model = FMModel(args.nfeat, args.nemb)
    elif args.model == 'hofm':
        model = HOFMModel(args.nfeat, args.nemb, args.k)
    elif args.model == 'afm':
        model = AFMModel(args.nfeat, args.nemb, args.h, args.dropout)
    elif args.model == 'dcn':
        model = CrossNetModel(args.nfield, args.nfeat, args.nemb, args.k)
    elif args.model == 'cin':
        model = CINModel(args.nfield, args.nfeat, args.nemb, args.k, args.h)
    elif args.model == 'afn':
        model = AFNModel(args.nfield, args.nfeat, args.nemb, args.h, args.nlayer, args.mlp_hid,
                    args.dropout, args.ensemble, args.dnn_nlayer, args.dnn_hid)
    elif args.model == 'armnet':
        model = ARMNetModel(args.nfield, args.nfeat, args.nemb, args.nattn_head, args.alpha, args.h,
                    args.nlayer, args.mlp_hid, args.dropout, args.ensemble, args.dnn_nlayer, args.dnn_hid)

    elif args.model == 'dnn':
        model = DNNModel(args.nfield, args.nfeat, args.nemb, args.nlayer, args.mlp_hid, args.dropout)
    elif args.model == 'gcn':
        model = GCNModel(args.nfield, args.nfeat, args.nemb, args.k, args.h, args.nlayer,
                         args.mlp_hid, args.dropout)
    elif args.model == 'gat':
        model = GATModel(args.nfield, args.nfeat, args.nemb, args.k, args.h,
                         args.nlayer, args.mlp_hid, args.dropout, 0.2, args.nattn_head)

    elif args.model == 'attreg':
        model = AttentiveRegModel(args.nfield, args.nfeat, args.nemb, args.k, args.h,
                         args.nlayer, args.mlp_hid, args.dropout, 0.2, args.nattn_head)
    elif args.model == 'attnoreg':
        model = AttentiveNoRegModel(args.nfield, args.nfeat, args.nemb, args.k, args.h,
                         args.nlayer, args.mlp_hid, args.dropout, 0.2, args.nattn_head)
    elif args.model == 'wd':
        model = WDModel(args.nfield, args.nfeat, args.nemb, args.nlayer, args.mlp_hid, args.dropout)
    elif args.model == 'ipnn':
        model = IPNNModel(args.nfield, args.nfeat, args.nemb, args.nlayer, args.mlp_hid, args.dropout)
    elif args.model == 'kpnn':
        model = KPNNModel(args.nfield, args.nfeat, args.nemb, args.nlayer, args.mlp_hid, args.dropout)
    elif args.model == 'nfm':
        model = NFMModel(args.nfeat, args.nemb, args.nlayer, args.mlp_hid, args.dropout)
    elif args.model == 'dfm':
        model = DeepFMModel(args.nfield, args.nfeat, args.nemb, args.nlayer, args.mlp_hid, args.dropout)
    elif args.model == 'dcn+':
        model = DCNModel(args.nfield, args.nfeat, args.nemb, args.k, args.nlayer, args.mlp_hid, args.dropout)
    elif args.model == 'xdfm':
        model = xDeepFMModel(args.nfield, args.nfeat, args.nemb, args.k, args.h,
                    args.nlayer, args.mlp_hid, args.dropout)

    elif args.model == 'fm_arm':
        model = FM_ARMModel(args.nfield, args.nfeat, args.nemb, args.nattn_head, args.alpha, args.h,
                            args.nlayer, args.mlp_hid, args.dropout, args.ensemble, args.dnn_nlayer, args.dnn_hid)

    else:
        raise ValueError(f'unknown model {args.model}')

    if torch.cuda.is_available(): model = model.cuda()
    logger.info(f'model parameters: {sum([p.data.nelement() for p in model.parameters()])}')
    return model
