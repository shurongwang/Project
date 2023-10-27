import os
from torch.nn import Embedding

# import argparse
# import os.path as osp

import torch
from torch import tensor
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

from torch import Tensor
from numpy import *

from matplotlib.pyplot import *

import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dim = 50
learning_rate = 0.0005
init = True

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=int, required=True)
parser.add_argument('--lr', type=float)
parser.add_argument('--init', type=int, required=True)
args = parser.parse_args()

dim = args.d
learning_rate = args.lr
init = args.init

print(f"Dimension: {dim:02d}, LR: {learning_rate:.5f}, init? {init}")

class Model(torch.nn.Module):
	
	def __init__(self, ent_n, rel_n, dim):
		super().__init__()
		
		self.ent_n = ent_n
		self.rel_n = rel_n
		self.dim = dim
		
		self.ent_emb = Embedding(ent_n, dim)
		self.rel_emb = Embedding(rel_n * (dim + 1), dim)
		
		# self.ent_emb.weight = F.sigmoid(self.ent_emb.weight)
	
	def init(self, ent_init_emb):
		self.ent_emb = Embedding.from_pretrained(ent_init_emb, freeze = False)
		# self.ent_emb.weight = torch.weight(ent_init_emb)
	
	def forward(self, h_ids, r_typ, t_ids):
		n = h_ids.numel()
		d = self.dim
		
		# print(f'n: {n}', h_ids, h_ids.numel())
		
		h = self.ent_emb(h_ids)
		t = self.ent_emb(t_ids)
		r_ids = r_typ * (d + 1)
		prod = torch.empty(n, 0, device = h_ids.device)
		
		# h = F.relu(h)
		# h = F.normalize(h)
		# h = h * 10
		# h = F.sigmoid(h)
		# h = F.relu(h)
		
		# t = F.relu(t)
		# t = F.normalize(t)
		# t = t * 10
		# t = F.sigmoid(t)
		# h = F.relu(h)
		
		# for k in range(d):
		# 	r = self.rel_emb(r_ids)
			
		# 	prod = torch.cat((prod, torch.sum(h * r, dim = 1).unsqueeze(0).transpose(0, 1)), dim = 1)
		# 	r_ids = r_ids + 1
			
		# r_off = self.rel_emb(r_ids)
		
		# prod = F.sigmoid(prod)
		# prod = prod + r_off
		
		rel = self.rel_emb(r_ids)
		r_ids = r_ids + 1
		off = self.rel_emb(r_ids)
		r_ids = r_ids + 1
		# coef = self.rel_emb(r_ids) / 4
		
		# rel = F.sigmoid(rel)
		
		# print(r_typ.expand_as(off))
		# print((r_typ.expand_as(off) * off))
		# print(h.size())
		
		offs = (r_typ + 1).unsqueeze(1).expand_as(off) * off
		# offs = off
		# prod = F.mish(h + rel - offs) + offs
		
		x = h + rel
		# prod = 1/8 * (x - offs)**2 + offs
		# prod = .5 * abs(x - offs) + offs
		prod = 1/1024 * (x - offs)**2 + offs
		
		# prod = F.tanh(rel)
		# prod = prod + r_off
		
		# prod = torch.relu(h * rel + r_off)
		
		# prod = F.relu(prod)
		# prod = F.normalize(prod)
		# prod = prod * 10
		
		return -F.cosine_similarity(prod, t)
		# return -torch.sum(prod * t) / prod.norm(dim = 1) / t.norm(dim = 1)
		# return (prod - t).norm(dim = 1)
		# return -(h + rel - t).norm(dim = -1)
	
	def loss(self, h_ids, r_typ, t_ids):
		# t_out = self.forward(h_ids, r_typ, t_ids)
		# return torch.sum(self.forward(h_ids, r_typ, t_ids))
		
		pos = self.forward(h_ids, r_typ, t_ids)
		neg = self.forward(*self.random_sample(h_ids, r_typ, t_ids))
		
		# h = self.ent_emb(h_ids)
		# t = self.ent_emb(t_ids)
		# gap = -F.cosine_similarity(h, t) / 4
		
		# return F.margin_ranking_loss(pos, neg, target = torch.ones_like(pos), margin = 1.0)
		
		# return 20 * torch.sum(pos) - torch.sum(neg)
		# return torch.sum(pos)
		return torch.sum(pos) + torch.sum(torch.abs(neg))# + torch.sum(gap)
	
	def random_sample(self, h_ids, r_typ, t_ids):
		# Random sample either `head_index` or `tail_index` (but not both):
		num_negatives = h_ids.numel() // 2
		rnd_ids = torch.randint(self.ent_n, h_ids.size(), device = h_ids.device)

		h_ids = h_ids.clone()
		h_ids[:num_negatives] = rnd_ids[:num_negatives]
		t_ids = h_ids.clone()
		t_ids[num_negatives:] = rnd_ids[num_negatives:]

		return h_ids, r_typ, t_ids
	
	def print_rel_emb(self, r_typ):
		d = self.dim
		r_ids = r_typ * (d + 1)
		
		rel = self.rel_emb(r_ids)
		print(rel)
		
		r_ids = r_ids + 1
		r_off = self.rel_emb(r_ids)
		print(r_off)


# model = Model(10, 2, 5)
# model.forward(torch.zeros([1], dtype = torch.long), torch.ones([1], dtype = torch.long), torch.zeros([1], dtype = torch.long))
# model.forward(torch.zeros([1, 2], dtype = torch.long), torch.ones([1, 2], dtype = torch.long), torch.zeros([1, 2], dtype = torch.long))

# set current work diectory
file_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(file_path)

ent_n = 14541
rel_n = 237
# ent_n = 16211
# rel_n = 14931

print(device)

model = Model(ent_n, rel_n, dim)
model.to(device)

raw = open("data/train.txt", "r").readlines()
train_h = []
train_r = []
train_t = []
for line in raw:
	h_txt, r_txt, t_txt = line.strip().split()
	h, r, t = int(h_txt), int(r_txt), int(t_txt)
	train_h.append(h)
	train_r.append(r)
	train_t.append(t)
train_h = tensor(train_h).to(device)
train_r = tensor(train_r).to(device)
train_t = tensor(train_t).to(device)

raw = open("data/valid.txt", "r").readlines()
valid_h = []
valid_r = []
valid_t = []
for line in raw:
	h_txt, r_txt, t_txt = line.strip().split()
	h, r, t = int(h_txt), int(r_txt), int(t_txt)
	valid_h.append(h)
	valid_r.append(r)
	valid_t.append(t)
valid_h = tensor(valid_h).to(device)
valid_r = tensor(valid_r).to(device)
valid_t = tensor(valid_t).to(device)

raw = open("data/test.txt", "r").readlines()
test_h = []
test_r = []
test_t = []
for line in raw:
	h_txt, r_txt, t_txt = line.strip().split()
	h, r, t = int(h_txt), int(r_txt), int(t_txt)
	test_h.append(h)
	test_r.append(r)
	test_t.append(t)
test_h = tensor(test_h).to(device)
test_r = tensor(test_r).to(device)
test_t = tensor(test_t).to(device)

M = []
for k in range(dim):
	M.append((torch.rand(size = (1, rel_n)) - 0.5) * 2)

if init:
	raw = open("init_emb.txt", "r").readlines()
	ent_init_emb = torch.empty(size = (ent_n, dim))
	ent_id = 0
	for line in raw:
		s = line.strip().split()
		for i in range(len(s)):
			s[i] = float(s[i])
		s = tensor(s)
		
		# if (s == 0).all():
		# 	print(ent_id)
		
		for k in range(dim):
			if k == 0:
				emb = torch.sum(M[k] * s).unsqueeze(0)
			else:
				emb = torch.cat((emb, torch.sum(M[k] * s).unsqueeze(0)))
		# print(emb)
		
		if (emb == 0).all():
			emb = (torch.rand(size = (1, dim)) - 0.5) * 2
		
		ent_init_emb[ent_id] = emb
		ent_id += 1
		
	print(ent_init_emb)

if (init): model.init(ent_init_emb)
# optimizer = optim.Adam(model.parameters(), lr = learning_rate)
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
scheduler1 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


# torch.autograd.set_detect_anomaly(True)

def train(h, r, t):
	n = h.numel()
	model.train()
	total_loss = 0
	
	batch_size = 1000
	chunk_h = train_h.split(batch_size)
	chunk_r = train_r.split(batch_size)
	chunk_t = train_t.split(batch_size)
	
	round = 0
	last = model.ent_emb.weight
	for (h, r, t) in zip(chunk_h, chunk_r, chunk_t):
		optimizer.zero_grad()
		loss = model.loss(h, r, t)
		loss.backward()
		optimizer.step()
		
		total_loss += float(loss)
		# print(model.ent_emb.weight)
		
		# round += 1
		# if (model.ent_emb.weight == last).all():
		# 	print('hhh', round)
		# last = model.ent_emb.weight
	return total_loss

@torch.no_grad()
def test(h_ids, r_typ, t_ids, log = True):
		model.eval()
		k = 10
		batch_size = 20000
		arange = range(h_ids.numel())
		arange = tqdm(arange) if log else arange
		
		rank_buck = [0] * ent_n
		
		MRR = 0.
		mean_ranks, hits_at_k = [], []
		for i in arange:
		# for i in range(1, 2):
			h, r, t = h_ids[i], r_typ[i], t_ids[i]
			
			# print(h, r, t)
			
			scores = []
			tail_indices = torch.arange(ent_n, device = h_ids.device)
			# j = 0
			for ts in tail_indices.split(batch_size):
				
				# print(ts)
				# print(model.forward(h.expand_as(ts), r.expand_as(ts), ts))
				# print()
				# print(i, j, model.forward(h.expand_as(ts), r.expand_as(ts), ts))
				# j += 1
				
				scores.append(model.forward(h.expand_as(ts), r.expand_as(ts), ts))
			
			# print(scores)
			
			# rank = int((torch.cat(scores).argsort(descending = True) == t).nonzero().view(-1))
			rank = int((torch.cat(scores).argsort(descending = False) == t).nonzero().view(-1))
			rank_buck[rank] += 1
			
			rk = rank + 1
			MRR += 1 / rk
			
			# all_scores = torch.cat(scores)
			# print(all_scores)
			# print(torch.cat(scores).argsort(descending = True))
			# exit()
			
			mean_ranks.append(rank)
			hits_at_k.append(rank < k)
		
		mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
		hits_at_k = int(torch.tensor(hits_at_k).sum()) / len(hits_at_k)
		MRR /= h_ids.numel()
		
		# for rk in range(1, ent_n + 1):
		# 	if (rank_buck[rk]):
		# 		print(rk, rank_buck[rk])
		
		print(f"{MRR: .4f}{mean_rank: .0f}{hits_at_k: .4f}")
		
		# plot(array(range(1, ent_n + 1)), array(rank_buck))
		# show()
		
		# return mean_rank, hits_at_k
		return

# if init: model.ent_emb.weight.requires_grad = False

total_epoch = 200
for epoch in range(1, total_epoch + 1):
	loss = train(train_h, train_r, train_t)
	print(f'Epoch {epoch:03d}, loss = {loss:.2f}')
	
	# print(model.ent_emb.weight)
	
	# if init and epoch > 50:
	# 	model.ent_emb.weight.requires_grad = True
	
	if (epoch % 25 == 0):
		test(valid_h, valid_r, valid_t)
		# model.print_rel_emb(tensor(range(0, rel_n)))
test(test_h, test_r, test_t)
# train(train_h, train_r, train_t)

# d = 5
# Loss = norm / 2, sigmoid on prod only, (e100, 1038, .2520)
# Loss = norm**.5, sigmoid on prod only, (e100, 1360, .2667)
# |p|				sigmod(prod)			(e025 .1360 1498 .2577)
# exp(norm)			sigmod(prod + off)		(e100 .1532 1389 .2688)
# |p| - sqrt(|n|)	sigmoid(prod + off)		(e150 .1504 1261 .2637)
# |p|				sigmod(prod) + off		(e075 .1526 1092 .2557)	use negative sampling!
# |p| - |n|			sigmod(prod) + off		(e225 .2028  904 .3341)
# second trial								(e025 .1943 1864 .3028)
# 											(e050 .1943 1277 .3227)
# 											(e075 .1932 1109 .3331)
# 											(e075 .2007 1036 .3361)
# 2|p| - |n|		sigmod(prod) + off		(e225 .1665  826 .2906)

# d = 5
# |p| - |n|			sigmoid(P) + off		(e025 .2166 1503 .3374)
#											(e050 .2248  946 .3571)
#											(e075 .2245  770 .3612)
#											(e100 .2279  690 .3595)
#											(e125 .2253  647 .3616)
#											(e150 .2219  618 .3571)
#											(e175 .2205  598 .3595)
#											(e200 .2172  583 .3616)
#											(ffff .2143  606 .3572)
#
# |p| - |n|			sigmoid(r) + off		(e025 .2106 1474 .3425)
#											(e050 .2174  953 .3546)
#											(e100 .2224  717 .3635)
#											(e150 .2065  647 .3599)
#											(e200 .2028  613 .3584)
#											(ffff .1993  627 .3515)
# lr = .0005
#											(e200 .2139  721 .3603)
#											(ffff .2113  756 .3543)
# d = 20									
#											(e200 .2201  540 .3624)
#											(ffff .2170  568 .3592)

# d = 10
# |p| - |n|			sigmoid(P) + off		(e025 .2326 1135 .3632)

# d = 50
# |p| - |n|			sigmoid(r) + off		(e025 .2374  921 .3708)

# 0.10917169084119162 695.3416748046875 0.17211291702309667
# 0.23288901964284742 501.87823486328125 0.3593384659252923

# d = 50, |p| - |n|, h + r
# ffff .2320  481 .3640

# inited, d = 5, |p| - |n|, h + r
# e025 .1858  784 .2914
# e050 .2248  538 .3507
# e075 .2304  503 .3587
# e100 .2326  495 .3620
# e125 .2335  496 .3640
# e150 .2344  497 .3640
# e175 .2339  501 .3648
# e200 .2332  504 .3647

# inited, d = 20, lr = 0.0002, cos, h + r
# e200 0.2285  405 0.3707
# ffff 0.2253  427 0.3671

# inited, d = 50, lr = 0.001, cos, h + r
# e200 0.2226  350 0.3628
# ffff 0.2191  373 0.3573