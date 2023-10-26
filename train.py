# # Analyze the knowledge graph a little bit, do it before training

import os
from torch.nn import Embedding

# import argparse
# import os.path as osp

import torch
from torch import tensor
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

# from torch_geometric.datasets import FB15k_237
# from torch_geometric.nn import ComplEx, DistMult, RotatE, TransE

from torch import Tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Model(torch.nn.Module):
	
	def __init__(self, ent_n, rel_n, dim):
		super().__init__()
		
		self.ent_n = ent_n
		self.rel_n = rel_n
		self.dim = dim
		
		self.ent_emb = Embedding(ent_n, dim)
		self.rel_emb = Embedding(rel_n * dim, dim)
		
		# self.ent_emb.weight = F.sigmoid(self.ent_emb.weight)
	
	def forward(self, h_ids, r_typ, t_ids):
		n = h_ids.numel()
		d = self.dim
		
		# print(f'n: {n}', h_ids, h_ids.numel())
		
		h = self.ent_emb(h_ids)
		t = self.ent_emb(t_ids)
		r_ids = r_typ * d
		prod = torch.empty(n, 0, device = h_ids.device)
		
		# h = F.relu(h)
		h = F.normalize(h)
		# h = h * 10
		# h = F.sigmoid(h)
		
		# t = F.relu(t)
		t = F.normalize(t)
		# t = t * 10
		# t = F.sigmoid(t)
		
		for k in range(d):
			r = self.ent_emb(r_ids)
			
			prod = torch.cat((prod, torch.sum(h * r, dim = 1).unsqueeze(0).transpose(0, 1)), dim = 1)
			r_ids = r_ids + 1
			
			# print(r_ids)
			# print()
			
		# prod = F.sigmoid(prod)
		# prod = F.relu(prod)
		prod = F.normalize(prod)
		# prod = prod * 10
		
		return (prod - t).norm(dim = 1)
		# return -(h + rel - t).norm(dim = -1)
	
	def loss(self, h_ids, r_typ, t_ids):
		# t_out = self.forward(h_ids, r_typ, t_ids)
		# return torch.sum(self.forward(h_ids, r_typ, t_ids))
		
		pos = self.forward(h_ids, r_typ, t_ids)
		neg = self.forward(*self.random_sample(h_ids, r_typ, t_ids))
		
		# return F.margin_ranking_loss(pos, neg, target = torch.ones_like(pos), margin = 1.0)
		
		return 20 * torch.sum(pos) - torch.sum(neg)
		# return torch.sum(pos) - torch
	
	def random_sample(self, h_ids, r_typ, t_ids):
		# Random sample either `head_index` or `tail_index` (but not both):
		num_negatives = h_ids.numel() // 2
		rnd_ids = torch.randint(self.ent_n, h_ids.size(), device = h_ids.device)

		h_ids = h_ids.clone()
		h_ids[:num_negatives] = rnd_ids[:num_negatives]
		t_ids = h_ids.clone()
		t_ids[num_negatives:] = rnd_ids[num_negatives:]

		return h_ids, r_typ, t_ids


# model = Model(10, 2, 5)
# model.forward(torch.zeros([1], dtype = torch.long), torch.ones([1], dtype = torch.long), torch.zeros([1], dtype = torch.long))
# model.forward(torch.zeros([1, 2], dtype = torch.long), torch.ones([1, 2], dtype = torch.long), torch.zeros([1, 2], dtype = torch.long))

# set current work diectory
file_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(file_path)

ent_n = 14513
rel_n = 237

model = Model(ent_n, rel_n, 50)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
# optimizer.to(device)

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

# torch.autograd.set_detect_anomaly(True)

def train(h, r, t):
	n = h.numel()
	model.train()
	total_loss = 0
	
	batch_size = 1000
	chunk_h = train_h.split(batch_size)
	chunk_r = train_r.split(batch_size)
	chunk_t = train_t.split(batch_size)
	
	for (h, r, t) in zip(chunk_h, chunk_r, chunk_t):
		optimizer.zero_grad()
		loss = model.loss(h, r, t)
		loss.backward()
		optimizer.step()
		
		total_loss += float(loss)
	return total_loss

@torch.no_grad()
def test(h_ids, r_typ, t_ids, log = True):
		model.eval()
		k = 10
		batch_size = 20000
		arange = range(h_ids.numel(), device = h_ids.device)
		arange = tqdm(arange) if log else arange
		
		mean_ranks, hits_at_k = [], []
		for i in arange:
		# for i in range(1, 2):
			h, r, t = h_ids[i], r_typ[i], t_ids[i]
			
			# print(h, r, t)
			
			scores = []
			tail_indices = torch.arange(ent_n, device = ent_n.device)
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
			
			# all_scores = torch.cat(scores)
			# print(all_scores)
			# print(torch.cat(scores).argsort(descending = True))
			# exit()
			
			mean_ranks.append(rank)
			hits_at_k.append(rank < k)
		
		mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
		hits_at_k = int(torch.tensor(hits_at_k).sum()) / len(hits_at_k)
		
		print(mean_rank, hits_at_k)
		
		return mean_rank, hits_at_k

# print(valid_h.numel())

# test(valid_h, valid_r, valid_t)

total_epoch = 500
for epoch in range(1, total_epoch + 1):
	loss = train(train_h, train_r, train_t)
	print(f'Epoch {epoch:03d}, loss = {loss:.2f}')
	
	if (epoch % 25 == 0):
		test(valid_h, valid_r, valid_t)
		
# train(train_h, train_r, train_t)
	