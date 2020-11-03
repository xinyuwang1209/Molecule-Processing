import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        x = torch.cat((x, ix), dim=1)

    return x




@torch.no_grad()
def sample(model, x, steps, temperature=1.0,boundary=None):
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
        logits, _ = model(x_cond,boundary=boundary)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, ix), dim=1)
    return x

'L_5*C(=O)NCc1cccc(OC)c1.*c1nsc2ccccc12COc1cccc(CNC(=O)c2cccc(NC(=O)c3nsc4ccccc34)c2)c1'

for i in range(1,21):
def sample_L(i,option='string'):
    # i=2
    prefix = 'L_'+str(i)
    string_input = prefix + '*O=C1NN=Cc2c1cccc2.*O=C(C1CC1)N1CCNCC1'
    array_input = [vocab[a] for a in ['<bos>'] + list(string_input)]
    boundary = [len(array_input)]
    tensor_input = torch.tensor(array_input,device='cuda').unsqueeze(0).repeat(32,1)
    boundary = boundary*32
    tensor_output = sample(model,tensor_input,250,boundary=boundary)
    strings_output = []
    for j in range(tensor_output.shape[0]):
        list_string_output = [inv[a] for a in tensor_output[j,boundary[j]:].cpu().numpy() if a != vocab['<pad>']]
        # if list_string_output[0] == '<bos>':
        #     list_string_output = list_string_output[1:]
        if list_string_output[-1] == '<eos>':
            list_string_output = list_string_output[:-1]
        string_output = ''.join(list_string_output)
        strings_output.append(string_output)
        print(string_output)
    for j in range(tensor_output.shape[0]):
        if test_valid(strings_output[j]):
            print(1)
        else:
            print(0)

    # logits,_ = model(tensor_input,boundary=boundary)


['<bos>', 'L', '_', '5', '*', 'C', '(', '=', 'O', ')', 'N', 'C', 'c', '1', 'c', 'c', 'c', 'c', '(', 'O', 'C', ')', 'c', '1', '.', '*', 'c', '1', 'n', 's', 'c', '2', 'c', 'c', 'c', 'c', 'c', '1', '2', 'C', 'O', 'c', '1', 'c', 'c', 'c', 'c', '(', 'C', 'N', 'C', '(', '=', 'O', ')', 'c', '2', 'c', 'c', 'c', 'c', '(', 'N', 'C', '(', '=', 'O', ')', 'c', '3', 'n', 's', 'c', '4', 'c', 'c', 'c', 'c', 'c', '3', '4', ')', 'c', '2', ')', 'c', '1', '<eos>']
