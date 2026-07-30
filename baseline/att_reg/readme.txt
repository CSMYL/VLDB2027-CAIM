(1) train_attreg_noreg.py

no att reg ....

(2) train_attreg_sep_grad.py

data grad and attreg grad seperately

(3) 21-7-11

adding reg_lambda for controlling regularization strength ...
reg_method for which way to calulcate reg strength
no value vectors ....

reg_lambda = reg_lambda * normalization_coefficient
---> since need to divided by normalization factor, so need to consider the off-set effects of normalization factor alsooooo!!!!
---> also has the batch size!!! ---> mean(dim=0)!!!  ---> no need, because it is mean!!!, not divided by batch_size

(4) 21-7-14

attentive_reg_tune_L2/ folder: using param group for all parameters ... + W using attreg, others use adam L2
attentive_reg_tune_L2_hidden_dot/ folder: copy from attentive_reg_tune_L2/ folder, but this uses h_i * h_j as attention ...

(5) 21-7-15

attentive_reg_tune_L2_hidden_dot: 

cosine similarity to replace dot product ...

### l2 normalization for cosine similarity
    xi_norm = xi / torch.norm(xi, dim=-1, keepdim=True)
    xj_norm = xj / torch.norm(xj, dim=-1, keepdim=True)
### l2 normalization for cosine similarity
    attn = torch.einsum('bfo,bfo->bf', xi_norm, xj_norm)

(6) 21-7-16 in attentive_reg_tune_L2_hidden_dot/

attn_method for different hidden dot, bilinear, attn_scalue, etc

(7) 21-7-16 in attentive_reg_tune_L2_hidden_dot/

use "model.gats[0].W" instead of "list(model_params['gats.0.W'])" !!!

otherwise not updated at all!!!

optimizer = optim.Adam([
            {'params': model.embedding.parameters(), 'weight_decay': args.wd},
            {'params': model.gats[0].W_bilinear, 'weight_decay': args.wd},
            {'params': model.gats[0].attn_scale, 'weight_decay': args.wd},
            {'params': model.gats[0].W_Q, 'weight_decay': args.wd},
            {'params': model.gats[0].W_Key, 'weight_decay': args.wd},
            {'params': model.gats[0].values, 'weight_decay': args.wd},
            {'params': model.gats[0].W, 'weight_decay': 0.0},
            # {'params': list(model_params['gats.1.W_Q']), 'weight_decay': 0.001},
            # {'params': list(model_params['gats.1.W_Key']), 'weight_decay': 0.001},
            # {'params': list(model_params['gats.1.values']), 'weight_decay': 0.001},
            # {'params': list(model_params['gats.1.W']), 'weight_decay': 0.0},
            {'params': model.affine.parameters(), 'weight_decay': args.wd}
            ],  lr=args.lr)

(8) solve NaN --- 21-7-19 in attentive_reg_tune_L2_hidden_dot/

(8-1)
need to divide xi and xj by abs(min) in order for underflow ...

### l2 normalization for cosine similarity
xi_min_abs = torch.min(torch.abs(xi), dim=-1, keepdim=True)[0]
xi_transform = xi / xi_min_abs                                           # in case of under flow ...
xj_min_abs = torch.min(torch.abs(xj), dim=-1, keepdim=True)[0]
xj_transform = xj / xj_min_abs                                           # in case of under flow ...

(8-2) 
even if the min(abs) can be zero ....

xi_min_abs = torch.where(xi_min_abs > 0.0, xi_min_abs, zero_min_abs)

(8-3)  7-20

xi_min_abs = torch.where(xi_min_abs > 1e-17, xi_min_abs, zero_min_abs)  ---> 1e-17

and float64


(9) 7-20

try attn * reg_strength, attn method: 11 ... (orthogonality)

(10) 7-23

attentive_reg_tune_L2_hidden_dot_uci: combine uci datasets

(11) 7-25

train_attreg.py --> adam for all model parameters, so if wd set to 0, then other params also set to zero
train_attreg_adam_attreg_only_attreg_w.py ---> param groups
train_attreg_only_w_uci.py --> integrate uci dataset inside

(12) 7-27

-- attreg adult can reach 0.8517 !!!
gat, attreg, attnreg, attwd all updated for uci datasets

-- calculate pearson correlation
train_attreg_only_w_uci_try_knwl.py: R2 = np.corrcoef(xarr, rowvar=False) --> each column is a variable
-- change uci_data_loader.py !!!

(13) 8-2 and 8-3 and 8-4

train_attnoreg_print_param_grad.py: try to print grad/param scale
train_print_param_grad.py: also, and no gradient clip ...
dict for saving dataset name and 150 steps grad/param ratio ..

(14) 8-25 SGD lr

attentive_reg_tune_L2_hidden_dot_uci/uci-generate-scripts: this folder contains the lr param/grad results ...

collect lr results ....

(15) 8-28 Adam lr

in attentive_reg_tune_L2_hidden_dot_uci/
try adam: train_attnoreg_print_param_grad.py and train_print_param_grad.py

(16) 8-29~8-30 -- Adam lr trying 0.01 and 0.001

lr + myadam try 0.01 and 0.001

(17) 9-4 try diff wds, adam, stratify=y, dataset name, lr, different wd configs

and no gradient clip for all train.py!!!
stratify=y  (may affect previous adult, movielens and frapple results ...)

(17-1) 9-4 --> 9-11 MLCask PM Demo

(18) 9-11: dataset name done + wd, gat done

(19) 9-13: fine-tune wd and reg_lambda for attreg

(20) 9-16: lazy update (folder: attentive_reg_hidden_dot_uci_test_lazy)
-- test where time is going 
-- lazy update strategy ...

(21) 9-17: try different methods of reducing training time --> 16-bit precision
optimizer.zero_grad()  needs to be moved ahead

(22) 9-18: write the improved cosine similarity -- $h_i$ $W$ $h_j$  --> attreg_two_lasyers.py
(attentive_reg_tune_L2_hidden_dot_uci/models)

(23) 9-21 ~ 9-26

interpretation code (run sf test code)

original code in "interpretation" folder

### in  attentive_reg_tune_L2_hidden_dot_uci
change my own code for interpretation ...  (train_attreg_only_w_uci_save_model.py)
-- save/load checkpoint
-- predict_proba
-- lime/shap + attreg
-- my attreg interpretation (train_attreg_only_w_uci_interpretation.py)

(24) 9-27 back to lazy update

attentive_reg_hidden_dot_uci_test_lazy/: old versions of train code + printing time ...

########## !!! attention, below use base_num = 0 or base_num > 0 to distinguish

(25) 9-28--10-2 base decomposition
attentive_reg_tune_L2_hidden_dot_uci_base/ 
--> actually just use base_num = 0 or base_num > 0 can distinguish

(26) 10-3--10-6 pruning + sparse weight

attentive_reg_tune_L2_hidden_dot_uci_base_sparse_weight/
-- attreg.py: self.h[i] * self.W[i] and self.W[i].norm(2) * self.W[i].norm(2) try only i%2 == 0 ....

attentive_reg_tune_L2_hidden_dot_uci_base_pytorch_pruning/
-- try pytorch pruning package ... -- both 'structured' and 'unstructured'
train_attreg_only_w_uci_pruning_time.py

Adam need to use W_orig as parameters ...

attentive_reg_tune_L2_hidden_dot_uci_base_group_lasso_prune/
group lasso + pruning--> attreg.py and attreg_two_layers.py 

(27) 10-7~10-9

collect all results

(28) 10-10~10-12

interpretation results ---> attentive_reg_tune_L2_hidden_dot_uci_base/train_attreg_only_w_uci_base_interpretation.py

(29) 10-13

shap global ...

###################### code memories #######################
attentive_reg_tune_L2_hidden_dot_uci: no bases
attentive_reg_tune_L2_hidden_dot_uci_base: num_bases=0 and num_bases>0 for Adam
###################### code memories #######################

(30) 22-1-17

--> replacing for loop using einsum!!!! --> test_autograd.py

(31) 22-1-18
print loss and accuracy for Avazu and Criteo also ...

(32) 22-1-26~1-27
einsum for all the for loops!!

(32-1) 22-1-27 ~ 2-19
Django for MLCask ...

(33) 22-2-19 ~ 22-2-22
check overfitting phenomenon using GAT model

(34) 22-3-13 ~ 3-14
tuning reg strength for displaying reg decreasing and no-reg increasing ..

(35) 22-3-16 ~ 3-17
efficiency attreg and attnoreg, attreg_two_layers --> simplified einsum!!!

(36) 22-3-25 ~ 22-4-15
SCUT YQ

(37) 22-4-18
seed_everything(args.seed)

(38) 22-4-17 ~ 4-22
debug opt einsum and otiginal einsum
4-22: leave the min/max for nan resloving alone ...

(39) 22-6-2 dataset deciding

(40) 22-6-7~22-6-8 data loader

(41) 22-6-9

41-1) gcn.py

41-2) train_attdropout.py
--> copying from train_attwd.py
--> but the graph weights have no weight decay --> Adam groups ...
--> normal graph model (no return of reg_loss, loss = loss + reg_loss ..NO)
--> reg is attnoregdrop2d --> copy from attnoreg

(42) 22-6-10~22-6-12: L1, L2, maxnorm, gradient clip, etc
change train_attdropout.py to train_baselines_only_w_uci.py
cp train_baselines_only_w_uci.py train_baselines_return_loss_only_w_uci.py --> used for L1 and L2 that need to be returned loss from model()
cp attnoreg.py attnoreg_lasso_wd.py --> this is used for L1 and L2 regularization added to the loss function

(43) 22-6-23: attentive_reg_tune_L2_hidden_dot_unified_base
copy from attentive_reg_tune_L2_hidden_dot_uci_base
all are libsvm + all are binary classification

(44) 22-6-29: attentive_reg_tune_L2_hidden_dot_unified_base_time
time measures .. 

(45) 22-7-1: train_attreg_only_w_uci_interpretation.py

(46) 22-10-3: knwl FDs code
(only in this train_attreg_only_w_uci.py file)

(47) 22-10-5: holoclean code integration
(48) 22-10-6: mask data loader, etc ...
(49) 22-10-11: knwl process FD + main function: 
train_attreg_only_w_uci_knwl.py (copy from train_attreg_only_w_uci.py) 
train_attreg_only_w_uci_tune_knwl.py --> tuning ....
