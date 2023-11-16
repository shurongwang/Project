# Convert the FB15k data set to dat that can be fed to the model

import os

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

raw = open("FB15k237/train.txt", "r").readlines()
for line in raw:
	h_txt, r_txt, t_txt = line.strip().split()
	e_dic.insert(h_txt)
	e_dic.insert(t_txt)
	r_dic.insert(r_txt)
raw = open("FB15k237/valid.txt", "r").readlines()
for line in raw:
	h_txt, r_txt, t_txt = line.strip().split()
	e_dic.insert(h_txt)
	e_dic.insert(t_txt)
	r_dic.insert(r_txt)
raw = open("FB15k237/test.txt", "r").readlines()
for line in raw:
	h_txt, r_txt, t_txt = line.strip().split()
	e_dic.insert(h_txt)
	e_dic.insert(t_txt)
	r_dic.insert(r_txt)

print(e_dic.size, r_dic.size)

wtr = open("triple.txt", "w")
raw = open("FB15k237/train.txt", "r").readlines()
for line in raw:
	h_txt, r_txt, t_txt = line.strip().split()
	h, r, t = e_dic.get_id(h_txt), r_dic.get_id(r_txt), e_dic.get_id(t_txt)
	wtr.write(str(h) + '\t' + str(r) + '\t' + str(t) + '\n')
raw = open("FB15k237/valid.txt", "r").readlines()
for line in raw:
	h_txt, r_txt, t_txt = line.strip().split()
	h, r, t = e_dic.get_id(h_txt), r_dic.get_id(r_txt), e_dic.get_id(t_txt)
	wtr.write(str(h) + '\t' + str(r) + '\t' + str(t) + '\n')
raw = open("FB15k237/test.txt", "r").readlines()
for line in raw:
	h_txt, r_txt, t_txt = line.strip().split()
	h, r, t = e_dic.get_id(h_txt), r_dic.get_id(r_txt), e_dic.get_id(t_txt)
	wtr.write(str(h) + '\t' + str(r) + '\t' + str(t) + '\n')
wtr.close()

wtr = open("entity2id.txt", "w")
for id in range(e_dic.size):
	txt = e_dic.txt[id]
	wtr.write(txt + '\t' + str(id) + '\n')
wtr.close()

wtr = open("relation2id.txt", "w")
for id in range(r_dic.size):
	txt = r_dic.txt[id]
	wtr.write(txt + '\t' + str(id) + '\n')
wtr.close()

# convert the training data
raw = open("FB15k237/train.txt", "r").readlines()
wtr = open("train.txt", "w")
for line in raw:
	h_txt, r_txt, t_txt = line.strip().split()
	h, r, t = e_dic.get_id(h_txt), r_dic.get_id(r_txt), e_dic.get_id(t_txt)
	wtr.write(str(h) + '\t' + str(r) + '\t' + str(t) + '\n')
wtr.close()

# convert the validation data
raw = open("FB15k237/valid.txt", "r").readlines()
wtr = open("valid.txt", "w")
for line in raw:
	h_txt, r_txt, t_txt = line.strip().split()
	h, r, t = e_dic.get_id(h_txt), r_dic.get_id(r_txt), e_dic.get_id(t_txt)
	wtr.write(str(h) + '\t' + str(r) + '\t' + str(t) + '\n')
wtr.close()

# convert the testing data
raw = open("FB15k237/test.txt", "r").readlines()
wtr = open("test.txt", "w")
for line in raw:
	h_txt, r_txt, t_txt = line.strip().split()
	h, r, t = e_dic.get_id(h_txt), r_dic.get_id(r_txt), e_dic.get_id(t_txt)
	wtr.write(str(h) + '\t' + str(r) + '\t' + str(t) + '\n')
wtr.close()
