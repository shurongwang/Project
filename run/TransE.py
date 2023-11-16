import os
import argparse

import torch
from torch import tensor
import torch.optim as optim
from torch.nn import Embedding
import torch.nn.functional as F

from tqdm import tqdm

from numpy import *
from matplotlib.pyplot import *

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--d', type = int)
parser.add_argument('--e', type = int)
parser.add_argument('--lr', type = float)
parser.add_argument('--i', type = int)
args = parser.parse_args()

# settings
dim 	= args.d if args.d != None else 			50
epoch_n = args.e if args.e != None else				200
lr 		= args.lr if args.lr != None else 			.0002
init 	= args.i if args.i != None else 			True

ent_n = 											14541	# -1 for auto
rel_n =												237		# -1 forauto

dirc = 												1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set current work diectory
file_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(file_path)

valid_triple = {}
dic = {}
class loader:
	def __init__ (self, prefix = "data/"):
		self.prefix = prefix
	
	def load(self, filename, add2dic = True):
		raw = open(self.prefix + filename).readlines()
		h_ids, r_typ, t_ids = [], [], []
		for line in raw:
			h_txt, r_txt, t_txt = line.strip().split()
			h, r, t = int(h_txt), int(r_txt), int(t_txt)
			if add2dic:
				valid_triple[(h, r, t)] = True
				if (h, r) not in dic:
					dic[(h, r)] = []
				dic[(h, r)].append(t)
			h_ids.append(h); r_typ.append(r); t_ids.append(t)
		h_ids = tensor(h_ids).to(device)
		r_typ = tensor(r_typ).to(device)
		t_ids = tensor(t_ids).to(device)
		return (h_ids, r_typ, t_ids)

# load data
loader = loader("data/")
tra_d = loader.load("train.txt")
val_d = loader.load("valid.txt")
tes_d = loader.load("test.txt")

if ent_n == -1: ent_n = int(max((max(tra_d[0]), max(tra_d[2]), max(val_d[0]), max(val_d[2]), max(tes_d[0]), max(tes_d[2])))) + 1
if rel_n == -1: rel_n = int(max((max(tra_d[1]), max(val_d[1]), max(tes_d[1])))) + 1

# print settings
print(f"using: {device}", sep = '')
print(f"# ent = {ent_n}, # rel type = {rel_n}", sep = '')
print(f"d: {dim:02d}, lr: {lr:.5f}, init? {init}", sep = '')

# model
class Model(torch.nn.Module):
	def __init__(self, ent_n, rel_n, dim, p = 2):
		super().__init__()
		
		self.ent_n = ent_n
		self.rel_n = rel_n
		self.dim = dim
		self.p = p

		# self.ent_emb = Embedding(ent_n, dim)
		# self.rel_emb = Embedding(rel_n * (dim + 1), dim)

		self.ent_emb = self._init_emb(ent_n)
		self.rel_emb = self._init_emb(rel_n)

	def _init_emb(self, num_embeddings):
		embedding = Embedding(num_embeddings=num_embeddings, embedding_dim=self.dim)
		uniform_range = 6 / np.sqrt(self.dim)
		embedding.weight.data.uniform_(-uniform_range, uniform_range)
		embedding.weight.data = torch.div(embedding.weight.data, embedding.weight.data.norm(p=2, dim=1, keepdim=True))
		
		return embedding

	def init(self, ent_init_emb):
		self.ent_emb = Embedding.from_pretrained(ent_init_emb, freeze = False)
		# self.rel_emb = Embedding.from_pretrained(rel_init_emb, freeze = False)
	
	def init_from_pretrained(self, ent_init_emb, rel_init_emb):
		self.ent_emb = Embedding.from_pretrained(ent_init_emb, freeze = False)
		self.rel_emb = Embedding.from_pretrained(rel_init_emb, freeze = False)

	def diff(self, v1, v2, d = -1):
		return (v1 - v2).norm(dim = d, p = self.p)

	def forward(self, h_ids, r_typ, t_ids):
		h = self.ent_emb(h_ids)
		t = self.ent_emb(t_ids)
		r = self.rel_emb(r_typ)
		
		# F.normalize(h)
		# F.normalize(r)
		# F.normalize(t)

		return self.diff(h + r, t)
	
	def loss(self, h_ids, r_typ, t_ids):
		# t_out = self.forward(h_ids, r_typ, t_ids)
		# return torch.sum(self.forward(h_ids, r_typ, t_ids))
		
		pos = self.forward(h_ids, r_typ, t_ids)
		neg = self.forward(*self.random_sample_2(h_ids, r_typ, t_ids))
		
		return F.margin_ranking_loss(pos, neg, target = -torch.ones_like(pos), margin = 1.)
		
	def random_sample_1(self, h_ids, r_typ, t_ids):
		num_negatives = h_ids.numel() // 2
		rnd_ids = torch.randint(self.ent_n, h_ids.size(), device = h_ids.device)

		h_ids = h_ids.clone()
		h_ids[:num_negatives] = rnd_ids[:num_negatives]
		t_ids = h_ids.clone()
		t_ids[num_negatives:] = rnd_ids[num_negatives:]

		return h_ids, r_typ, t_ids

	def random_sample_2(self, h_ids, r_typ, t_ids):
		h_or_t = torch.randint(1, h_ids.size(), device = device)
		rnd_ids = torch.randint(self.ent_n, h_ids.size(), device = device)
		h_neg = torch.where(h_or_t == 1, rnd_ids, h_ids)
		t_neg = torch.where(h_or_t == 0, rnd_ids, t_ids)

		return h_neg, r_typ, t_neg

	def print_rel_emb(self, r_typ):
		d = self.dim
		r_ids = r_typ * (d + 1)
		
		rel = self.rel_emb(r_ids)
		print(rel)
		
		r_ids = r_ids + 1
		r_off = self.rel_emb(r_ids)
		print(r_off)

def train(data, batch_s = 2000):
	model.train()
	total_loss = 0
	
	n = data[0].numel()
	(h_ids, r_typ, t_ids) = data

	shuffled_ids = torch.randperm(n)
	h_ids = h_ids[shuffled_ids]
	r_typ = r_typ[shuffled_ids]
	t_ids = t_ids[shuffled_ids]
	
	h_ids = h_ids.split(batch_s); r_typ = r_typ.split(batch_s); t_ids = t_ids.split(batch_s)
	for d in zip(h_ids, r_typ, t_ids):
		optimizer.zero_grad()

		loss = model.loss(d[0], d[1], d[2])
		loss.backward()
		optimizer.step()
		
		total_loss += float(loss)

		F.normalize(model.ent_emb.weight.data)
		F.normalize(model.rel_emb.weight.data)
			
	return total_loss

@torch.no_grad()
def eval_batch_wise(h_ids, r_typ, t_ids):
	test_size = h_ids.numel()

	h_all = h_ids.repeat_interleave(ent_n)
	r_typ = r_typ.repeat_interleave(ent_n)
	t_all = torch.arange(ent_n, device = device).repeat(test_size)

	score = dirc * model.forward(h_all, r_typ, t_all).view(-1, ent_n)
	_, indices = torch.topk(score, k = model.ent_n, dim = 1, largest = False)
	
	tail = t_ids.view(-1, 1)
	rank = torch.eq(indices, tail).nonzero().permute(1, 0)[1] + 1
	MR = torch.sum(rank)
	MRR = torch.sum(1 / rank)
	H_at_1 = torch.sum(torch.eq(indices[:, :1], tail)).item()
	H_at_3 = torch.sum(torch.eq(indices[:, :3], tail)).item()
	H_at_10 = torch.sum(torch.eq(indices[:, :10], tail)).item()
	return MR, MRR, H_at_1, H_at_3, H_at_10

@torch.no_grad()
def eval(data, colo = "green", batch_s = 2000):
	model.eval()
	
	n = data[0].numel()
	MR, MRR, H_at_1, H_at_3, H_at_10 = 0., 0., 0., 0., 0.
	
	(h_ids, r_typ, t_ids) = data
	h_ids = h_ids.split(batch_s); r_typ = r_typ.split(batch_s); t_ids = t_ids.split(batch_s)
	
	m = len(h_ids)	
	# for d in zip(h_ids, r_typ, t_ids):
	for _ in tqdm(range(m), colour = colo):
		h, r, t = h_ids[_], r_typ[_], t_ids[_]
		# h, r, t = d[0], d[1], d[2]
		a1, a2, a3, a4, a5 = eval_batch_wise(h, r, t)
		MR += a1; MRR += a2; H_at_1 += a3; H_at_3 += a4; H_at_10 += a5
	
	MR /= n; MRR /= n; H_at_1 /= n; H_at_3 /= n; H_at_10 /= n
	print(f"MR{MR: .0f} MRR{MRR: .4f} H@1{H_at_1: .4f} H@3{H_at_3: .4f} H@10{H_at_10: .4f}", sep = '')
	
	return MR, MRR, H_at_1; H_at_3; H_at_10

@torch.no_grad()
def eval_with_filter(data, colo = "green", batch_s = 2000):
	model.eval()
	model_device = next(model.parameters()).device
	# model.to(device)
	
	n = data[0].numel()
	MR, MRR, H_at_1, H_at_3, H_at_10 = 0., 0., 0., 0., 0.
	
	for i in tqdm(range(n), colour = colo):
		h, r, t = data[0][i], data[1][i], data[2][i]
		
		ents = torch.arange(ent_n)
		cur = (int(h), int(r)); ans = int(t)
		debuff = torch.zeros_like(ents, device = device)
		if cur in dic:
			for k in dic[cur]:
				if k != ans:
					# ents[k] = t
					debuff[k] = 1e9
		
		ents = ents.to(model_device)
		score = model.forward(h.expand_as(ents), r.expand_as(ents), ents)
		
		rank = int(((score + debuff).argsort(descending = False) == t).nonzero().view(-1)) + 1
		
		# assert(rank != -1)
		MR += rank; MRR += 1 / rank
		H_at_1 += rank <= 1; H_at_3 += rank <= 3; H_at_10 += rank <= 10

	MR /= n; MRR /= n; H_at_1 /= n; H_at_3 /= n; H_at_10 /= n
	print(f"MR{MR: .0f} MRR{MRR: .4f} H@1{H_at_1: .4f} H@3{H_at_3: .4f} H@10{H_at_10: .4f}", sep = '')
	
	model.to(model_device)
	return

def expand(data):
	n = len(data)
	h = data[0].clone()
	r = data[1].clone()
	t = data[2].clone()

	# print(torch.cat((data[1], r + rel_n)))
	# print(data[0].numel(), "lines")
	return (h, r, t)
	# return (t, r, h)
	# return (torch.cat((data[0], t)), torch.cat((data[1], r + rel_n)), torch.cat((data[2], h)))

tra_d = expand(tra_d)
val_d = expand(val_d)
tes_d = expand(tes_d)

M = []
for k in range(dim):
	M.append((torch.rand(size = (1, rel_n), device = device) - 0.5) * 2)

if init == 1:
	raw = open("init_emb.txt", "r").readlines()
	ent_init_emb = torch.empty(size = (ent_n, dim), device = device)
	# rel_init_emb = torch.empty(size = (rel_n, dim), device = device)
	
	M = []
	for k in range(dim):
		M.append((torch.rand(size = (1, rel_n), device = device) - 0.5) * 2)

	ent_id = 0
	for line in tqdm(raw):
		s = line.strip().split()
		for i in range(len(s)):
			s[i] = float(s[i])
		s = tensor(s, device = device)
		
		ent_init_emb[ent_id] = s; ent_id += 1

	#	for k in range(dim):
	#		emb = torch.sum(M[k] * s).unsqueeze(0) if k == 0 else torch.cat((emb, torch.sum(M[k] * s).unsqueeze(0)))
		
	#	if (emb == 0).all(): emb = (torch.rand(size = (1, dim), device = device) - 0.5) * 2

	#	ent_init_emb[ent_id] = emb; ent_id += 1
	
	ent_init_emb = F.normalize(ent_init_emb)
	print("init done")

model = Model(ent_n, rel_n, dim, 2)
model.to(device)

if init == 1:
	model.init(ent_init_emb)
	model.ent_emb.weight.requires_grad = True
	model.rel_emb.weight.requires_grad = True
elif init == 2:
	ent_emb = torch.load('ent_emb.pth'); ent_emb.to(device)
	rel_emb = torch.load('rel_emb.pth'); rel_emb.to(device)
	
	print(ent_emb.shape, rel_emb.shape)

	model.init_from_pretrained(ent_emb, rel_emb)
	
optimizer = optim.Adam(model.parameters(), lr = lr)

for epoch in range(1, epoch_n + 1):
	loss = train(tra_d)
	# print(f'Epoch {epoch:03d}, loss = {loss:.2f}')
	
	if init and epoch > 50:
		model.ent_emb.weight.requires_grad = True
	
	if (epoch % 25 == 0):
		print(f"Epoch: {epoch:03d}, Loss = {loss:.2f}", sep = '')
		eval_with_filter(val_d)
		eval_with_filter(tes_d)

eval_with_filter(val_d, colo = 'red')
eval_with_filter(tes_d, colo = 'blue')

store_emb = 1

if (store_emb):
	ent_emb = model.ent_emb(torch.arange(ent_n, device = device))
	torch.save(ent_emb, "ent_emb.pth")
	rel_emb = model.rel_emb(torch.arange(rel_n, device = device))
	torch.save(rel_emb, "rel_emb.pth")
