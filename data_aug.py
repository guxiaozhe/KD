import torch 
import  numpy as np
from torchray.attribution.gradient import gradient
# import pudb
from numpy.random import beta

def one_hot(x, n_class, dtype=torch.float32): 
    # X shape: (batch), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res


class Augmenter():
	def __init__(self, config):
		self.config = config
	def augdata(self, input,target):
		return input,target

		



class Mixup(Augmenter):
	def __init__(self,config):
		super(Mixup, self).__init__(config)
		self.alpha=config["alpha"]
		self.n_class=config["n_class"]
		self.aug_prob=config["aug_prob"]
	def augdata(self, input,target):
		r = np.random.rand(1)
		target=one_hot(target,self.n_class)
		if  r <self.aug_prob:
			lam= beta(self.alpha,  self.alpha)
			rand_index = torch.randperm(input.size(0))
			perm_input = input[rand_index]
			perm_target = target[rand_index]
			return input.mul_(lam).add_(1 - lam, perm_input), target.mul_(lam).add_(1 - lam, perm_target)
		else:
			return input, target




class CutMix(Augmenter):
	def __init__(self,config):
		super(CutMix, self).__init__(config)
		self.alpha=config["alpha"]
		self.n_class=config["n_class"]
		self.aug_prob=config["aug_prob"]

	def augdata(self, input,target):
		r = np.random.rand(1)
		target=one_hot(target,self.n_class)
		if r <self.aug_prob:
			# generate mixed sample
			lam= beta(self.alpha,  self.alpha)
			rand_index = torch.randperm(input.size(0))
			target_a = target
			target_b = target[rand_index]
			bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
			input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
			# adjust lambda to exactly match pixel ratio
			lam_t = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
			return input, target_a*lam_t+target_b * (1. - lam_t)
		else:
			return input, target


class CutMixKD(Augmenter):
	def __init__(self,config):
		super(CutMixKD, self).__init__(config)
		self.alpha=config["alpha"]
		self.n_class=config["n_class"]
		self.aug_prob=config["aug_prob"]
		self.t_net=config["t_net"]
		self.device=config["device"]

	def augdata(self, input,target):
		r = np.random.rand(1)
		target=one_hot(target,self.n_class)
		if r <self.aug_prob:
			lam= beta(self.alpha,  self.alpha)
			rand_index = torch.randperm(input.size(0))
			target_a = target
			target_b = target[rand_index]
			bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
			input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
			n_sample=input.size(0)
			saliency = gradient(self.t_net, input, target)
			smax=saliency.view( n_sample,-1).sum(dim=1)
			mask=torch.ones(saliency.shape)
			mask=mask.to(self.device)
			mask[:,:,bbx1:bbx2, bby1:bby2]=0
			sa=(saliency*mask).view(n_sample,-1).sum(dim=1)/smax
			sb=saliency[rand_index, :, bbx1:bbx2, bby1:bby2].view( n_sample,-1).sum(dim=1)/smax[rand_index]
			lam_t=sa/(sa+sb)
			# pudb.set_trace()
			lam_t=(lam_t.unsqueeze(1)).expand(n_sample,self.n_class)
			return input, target_a*lam_t+target_b * (1. - lam_t)
		else:
			return input, target




def rand_bbox(size, lam):
	W = size[2]
	H = size[3]
	cut_rat = np.sqrt(1. - lam)
	cut_w = np.int(W * cut_rat)
	cut_h = np.int(H * cut_rat)

	# uniform
	cx = np.random.randint(W)
	cy = np.random.randint(H)

	bbx1 = np.clip(cx - cut_w // 2, 0, W)
	bby1 = np.clip(cy - cut_h // 2, 0, H)
	bbx2 = np.clip(cx + cut_w // 2, 0, W)
	bby2 = np.clip(cy + cut_h // 2, 0, H)

	return bbx1, bby1, bbx2, bby2



