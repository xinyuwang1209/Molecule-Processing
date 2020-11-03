def convert2fgs(molecule):
    keys = MACCSkeys.GenMACCSKeys(molecule)
    keys = np.array(keys.GetOnBits())
    fingerprint = np.zeros(166, dtype='uint8')
    if len(keys) != 0:
        fingerprint[keys - 1] = 1
    return fingerprint

def get_SNN(stock_vecs,gen_vecs,size_batch=2048,agg='max',device='cuda',p=1):
    agg='max'
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], size_batch):
        x_stock = torch.tensor(stock_vecs[j:j + size_batch]).to(device).float()
        for i in range(0, gen_vecs.shape[0], size_batch):
            y_gen = torch.tensor(gen_vecs[i:i + size_batch]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                         y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac**p
            if agg == 'max':
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1/p)
    return np.mean(agg_tanimoto)


def process_df(df):
    df[VALIDITY] = df[SMILE].apply(test_valid)
    df[IS_VALID] = ~df[VALIDITY].isnull()
    df[IS_NOVAL] = ~df[SMILE].isin(df_train_raw[SMILE])
    df_valid = df.loc[df[IS_VALID]][[SMILE,VALIDITY]]
    df_valid['CANONIC'] = df_valid[VALIDITY].apply(lambda x: Chem.MolToSmiles(x))
    df_valid['IS_UNIQUE'] = ~df_valid['CANONIC'].duplicated()
    n_novalty = df[IS_NOVAL].sum()
    n_valid = df[IS_VALID].sum()
    n_unique_k = df_valid['IS_UNIQUE'].iloc[:unique_k].sum()
    return df,df_valid,n_novalty,n_valid,n_unique_k


def sample_mask(dim,size_batch,mask_pading=False,device='cpu'):
    mask = (torch.triu(torch.ones(dim, dim,dtype=torch.long,device=device)) == 1).transpose(0, 1)
    mask = mask.repeat(size_batch,1,1)
    if mask_pading:
        for i in range(size_batch):
            mask[i,:,maxlength_batch-n_pad[i]:] = False
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
# def get_fingerprints(df):
#     df['fingerprint_maccs']=df[VALIDITY].apply(convert2fgs)
#     length = 1
#     for index_test_valid_unique in range(df.shape[0]):
#         fp = df.iloc[index_test_valid_unique]['fingerprint_maccs']
#         if fp is not None:
#             length = fp.shape[-1]
#             first_fp = fp
#             break
#     df['fingerprint_maccs']=df['fingerprint_maccs'].apply(lambda fp: fp if fp is not None else np.array([np.NaN]).repeat(length)[None, :])
#     return np.vstack(df['fingerprint_maccs'])

def get_fingerprints(df,morgan__r=2,morgan__n=1024):
    df['fingerprint_maccs']=df[VALIDITY].apply(convert2fgs)
    length = 1
    for index_test_valid_unique in range(df.shape[0]):
        fp = df.iloc[index_test_valid_unique]['fingerprint_maccs']
        if fp is not None:
            length = fp.shape[-1]
            first_fp = fp
            break
    df['fingerprint_maccs']=df['fingerprint_maccs'].apply(lambda fp: fp if fp is not None else np.array([np.NaN]).repeat(length)[None, :])
    return np.vstack(df['fingerprint_maccs'])

def sample_AMG(model,vocab,inverted_vocab,size_batch=2048,maxlength_sequence=120,device='cuda',temperature=1.):
    x = torch.tensor([[vocab['<bos>']]*size_batch], dtype=torch.long,device=device).transpose(0,1).contiguous() # context conditioning
    y = sample(model, x, steps=maxlength_sequence, temperature=temperature, sample=True, top_k=None)
    list_smiles = [''.join([inverted_vocab[b.item()] for b in y[i,1:] if b != vocab['<pad>']]) for i in range(y.shape[0])]
    list_smiles = [a.split('<eos>')[0] for a in list_smiles]
    return list_smiles


def sample_n_AMG(n_to_sample,size_batch,parameters_sample):
    # size_batch = parameters_sample.size_batch
    sampled = []
    while len(sampled)<n_to_sample:
        size_batch_current = min(n_to_sample-len(sampled),size_batch)
        list_string_sampled = sample_AMG(size_batch=size_batch_current,**parameters_sample)
        # print(i,times)
        # current_string = (''.join([inverted_vocab[a] for a in tensor_sampled[1:,i].cpu().numpy() if inverted_vocab[a]!='<pad>'])).split('<eos>')
        sampled += list_string_sampled
        print(len(sampled))
    return sampled


def sample_n(n_to_sample,inverted_vocab,size_batch,parameters_sample,sample):
    # size_batch = parameters_sample.size_batch
    sampled = []
    while len(sampled)<n_to_sample:
        size_batch_current = min(n_to_sample-len(sampled),size_batch)
        tensor_sampled = sample(**parameters_sample)
        tensor_sampled = tensor_sampled[:,:size_batch_current]
        for i in range(tensor_sampled.shape[0]):
            current_string = [inverted_vocab[a] for a in tensor_sampled[i,1:].cpu().numpy() if a in inverted_vocab.keys()]
            sampled.append(current_string)
    return sampled

def array2string(inverted_vocab,path,new_name):
    df = torch.load(path)
    df[0] = df[0].apply(lambda x: x[1:-1])
    df[0] = df[0].apply(lambda x: ''.join([inverted_vocab[a] for a in x]))
    df.columns = ['SMILES']
    torch.save(df,new_name)
    return df
