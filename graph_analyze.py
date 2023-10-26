# Analyze the knowledge graph a little bit, do it before training

import os
import queue

# set current work diectory
file_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(file_path)

class dictionary:
	def __init__(self):
		self.size = 0
		self.dic = {}
		self.txt = []
	
	def insert(self, s):
		if s not in self.dic:
			self.dic[s] = self.size
			self.txt.append(s)
			self.size += 1
	
	def get_id(self, s):
		if s not in self.dic:
			return -1
		return self.dic[s]

e_dic = dictionary()
r_dic = dictionary()

class edge:
	def __init__(self, h, r, t):
		self.h = h
		self.r = r
		self.t = t
		self.id = id

class knowledge_graph:
	def __init__(self, node_size):
		self.node_size = node_size
		self.edge_size = 0
		self.e = []
		self.adj = [[] for _ in range(node_size)]
		self.dic = {}
	
	def insert(self, h, r, t):
		id = self.edge_size
		self.e.append(edge(h, r, t))
		self.adj[h].append(id)
		self.dic[(h, r, t)] = id
		self.edge_size += 1
	
	def get_id(self, h, r, t):
		if (h, r, t) not in self.dic:
			return -1
		return self.dic[(h, r, t)]

raw = open("data/entity2id.txt", "r").readlines()
for line in raw:
	e_txt, id = line.strip().split()
	e_dic.insert(e_txt)

raw = open("data/relation2id.txt", "r").readlines()
for line in raw:
	r_txt, id = line.strip().split()
	r_dic.insert(r_txt)

entity_count = e_dic.size
relation_count = r_dic.size

print(entity_count, relation_count)

G = knowledge_graph(entity_count)
rG = knowledge_graph(entity_count)

rel = [[] for _ in range(relation_count)]

raw = open("data/triple.txt", "r").readlines()
for line in raw:
	h, r, t = line.strip().split()
	h = int(h); r = int(r); t = int(t)
	G.insert(h, r, t)
	rG.insert(t, r, h)
	rel[r].append(G.get_id(h, r, t))

def get_path(G, u, pre1, pre2):
	h = u; r1 = []
	while pre1[h] != -1:
		e_id = pre1[h]
		# r1.append(e_id)
		r1.append(G.e[e_id].r)
		h = G.e[e_id].h
	t = u; r2 = []
	while pre2[t] != -1:
		e_id = pre2[t]
		# r2.append(e_id)
		r2.append(G.e[e_id].r)
		t = G.e[e_id].t
	r1.reverse()
	r = r1 + r2
	# print(r1)
	# print(r2)
	# print()
	return (h, r, t)

hits = 10
def bfs(G, rG, s, t, ban_rel):
	n = G.node_size
	q1 = queue.Queue(); vis1 = [0] * n; pre1 = [0] * n
	q2 = queue.Queue(); vis2 = [0] * n; pre2 = [0] * n
	path_list = []
	
	q1.put(s); vis1[s] = 1; pre1[s] = -1
	q2.put(t); vis2[t] = 1; pre2[t] = -1
	while len(path_list) < hits and (not q1.empty() or not q2.empty):
		# if ban_id == 67320: print(path_list, q1.qsize(), q2.qsize())
		if q1.qsize() > q2.qsize():
			u = q1.get()
			if vis2[u]:
				path = get_path(G, u, pre1, pre2)
				found = 0
				for p in path_list:
					if p == path:
						found = 1
				if not found:
					path_list.append(path)
			for e_id in G.adj[u]:
				if G.e[e_id].r == ban_rel: continue
				v = G.e[e_id].t
				if not vis1[v]:
					q1.put(v); vis1[v] = 1; pre1[v] = e_id
		else:
			u = q2.get()
			if vis1[u]:
				path = get_path(G, u, pre1, pre2)
				found = 0
				for p in path_list:
					if p == path:
						found = 1
				if not found: 
					path_list.append(path)
			for e_id in rG.adj[u]:
				if rG.e[e_id].r == ban_rel: continue
				v = rG.e[e_id].t
				if not vis2[v]:
					q2.put(v); vis2[v] = 1; pre2[v] = e_id
	return path_list

dick = {}

def list2str(a):
	s = ""
	for u in a:
		s += str(u) + ','
	return s

# for r in range(relation_count):
for r in range(2, 3):
	i = 0
	for e_id in rel[r]:
		h, t = G.e[e_id].h, G.e[e_id].t
		path = bfs(G, rG, h, t, r)
		# print()
		# print(path)
		i += 1
		print(i, '/', len(rel[r]), e_id)
		# if e_id:
		# 	break
		
		# if i == 200:
		# 	break
		
		for p in path:
			(h, rr, t) = p
			# print("r:", list2str(rr))
			s = list2str(rr)
			if s not in dick:
				dick[s] = 0
			dick[s] += 1
		
	if r == 1:
		break

d = []
for key in dick:
	value = dick[key]
	d.append((value, key))
d.sort()
d.reverse()
i = 0
for (value, key) in d:
	print(value, key)
	i += 1
	if i == 20:
		break

# 72, 157, 58

# r = [209973, 67336, 158067, 143240]

# u = 1712
# for e_id in r:
# 	h = G.e[e_id].h
# 	r = G.e[e_id].r
# 	t = G.e[e_id].t
	
# 	u = G.e[e_id].t
# 	print(e_id, h, r, t)

# print(u)

# what about prefix and suffix?