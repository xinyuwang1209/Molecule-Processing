from SMILE_Generation.utils.utils        import *
from SMILE_Generation.utils.utils_sample  import *
import torch.nn.functional as F

def sample(model,vocab_bos,size_batch=32,size_block=70,temperature=1.,):
    model,device = load_to_device(model)
    model.eval()
    with torch.no_grad():
        tensor_sampled = torch.zeros(size_batch,size_block+1,dtype=torch.long,device=device)
        tensor_sampled[:,0] = vocab_bos
        hiddens = None
        for i in range(size_block):
            input_current = tensor_sampled[:,[i]]
            probs,hiddens = model.forward(input_current,hiddens)
            probs = probs[:,-1]
            probs = probs * temperature
            probs = F.softmax(probs,dim=-1)
            sample = torch.distributions.categorical.Categorical(probs).sample()
            tensor_sampled[:,i+1] = sample
        return tensor_sampled
