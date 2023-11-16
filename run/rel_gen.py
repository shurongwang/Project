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
dim 	= args.d if args.d != None else 			100
epoch_n = args.e if args.e != None else				2000
lr 		= args.lr if args.lr != None else 			.0002
init 	= args.i if args.i != None else 			True

ent_n = 											237
rel_n =												0

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set current work diectory
file_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(file_path)

class loader:
	def __init__ (self, prefix = "data/"):
		self.prefix = prefix
	
	def load(self, filename):
		raw = open(self.prefix + filename).readlines()
		h_ids, r_typ, t_ids = [], [], []
		for line in raw:
			h_txt, r_txt, t_txt = line.strip().split()
			h, r, t = int(h_txt), int(r_txt), int(t_txt)
			h_ids.append(h); r_typ.append(r); t_ids.append(t)
		h_ids = tensor(h_ids).to(device)
		r_typ = tensor(r_typ).to(device)
		t_ids = tensor(t_ids).to(device)
		return (h_ids, r_typ, t_ids)

# load data
# loader = loader("data/")
# tra_d = loader.load("train.txt")
# val_d = loader.load("valid.txt")
# tes_d  = loader.load("test.txt")

# if ent_n == -1: ent_n = int(max((max(tra_d[0]), max(tra_d[2]), max(val_d[0]), max(val_d[2]), max(tes_d[0]), max(tes_d[2])))) + 1
# if rel_n == -1: rel_n = int(max((max(tra_d[1]), max(val_d[1]), max(tes_d[1])))) + 1

# print settings
print(f"using: {device}", sep = '')
print(f"# ent = {ent_n}, # rel type = {rel_n}", sep = '')
print(f"d: {dim:02d}, lr: {lr:.5f}, init? {init}", sep = '')

# model
class Model(torch.nn.Module):
	def __init__(self, ent_n, dim):
		super().__init__()
		
		self.ent_n = ent_n
		self.dim = dim
		
		self.ent_emb = Embedding(ent_n, dim)
	
	def forward(self, h_ids, t_ids):
		h = self.ent_emb(h_ids)
		t = self.ent_emb(t_ids)
		
		F.normalize(h)
		F.normalize(t)
		
		w = F.cosine_similarity(h, t)
		w = w.where(h_ids != t_ids, tensor(0., device = device))

		return w
	
	def loss(self, h_ids, t_ids):
		norms = self.ent_emb.weight.data.norm(dim = 1)
		
		return torch.max(self.forward(h_ids, t_ids))

def train(batch_s = 60000):
	model.train()
	total_loss = 0
	
	h_ids = torch.arange(ent_n, device = device).repeat_interleave(ent_n)
	t_ids = torch.arange(ent_n, device = device).repeat(ent_n)
	
	h_ids = h_ids.split(batch_s); t_ids = t_ids.split(batch_s)
	for d in zip(h_ids, t_ids):
		optimizer.zero_grad()
		loss = model.loss(d[0], d[1])
		loss.backward()
		optimizer.step()
		
		total_loss += float(loss)
	
	return total_loss

model = Model(ent_n, dim)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr = lr)

h_ids = torch.arange(ent_n, device = device).repeat_interleave(ent_n)
t_ids = torch.arange(ent_n, device = device).repeat(ent_n)	

print("before:", float(model.loss(h_ids, t_ids)))

for epoch in range(1, epoch_n + 1):
	loss = train()
	if epoch % 1000 == 0:
		print(f'Epoch {epoch:04d}, loss = {loss:.2f}')

norms = model.forward(h_ids, t_ids)

print(norms)

S1, S2, S3, S4, cnt = 0., 0., 0., 0., 0.
M = model.ent_emb.weight.data

print(M[0])

wtr = open("M.txt", "w")
dat = ''

for i in range(ent_n):
	u = M[i]
	S1 += torch.norm(u)

	s = ''
	for j in range(dim):
		s += str(float(u[j])) + ' '
	dat += s + '\n'

	for j in range(i + 1, ent_n):
		v = M[j]

		l = torch.norm(u - v)
		
		S4 += torch.sum(u * v)

		S3 += l**2

		S2 += l
		cnt += 1
		# print(f"{l: .2f}", end = ' ')
	# print()
S1 /= ent_n
S2 /= cnt
S3 /= cnt
S4 /= cnt
print("len:", S1)
print("diff:", S2)
print("StdDev:", S3 - S2**2)
print("CosSim:", S4)

wtr.write(dat)
wtr.close()
