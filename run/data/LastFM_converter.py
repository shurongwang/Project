# Convert the LastFM data set to dat that can be fed to the model

import os
from numpy import *

# set current work diectory
file_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(file_path)

ent_n, rel_n = 0, 0

raw = open("LastFM/triple.txt", "r").readlines()
head, relation, tail = [], [], []
n = len(raw)
for line in raw:
	h, r, t = line.strip().split()
	h, r, t = int(h), int(r), int(t)
	head.append(h); relation.append(r); tail.append(t)
	
	ent_n = max(ent_n, h)
	ent_n = max(ent_n, t)
	rel_n = max(rel_n, r)


train = []
valid = []
test = []

for i in range(n):
	h, r, t = head[i], relation[i], tail[i]
	p = random.rand()
	if p > 0.9:
		test.append((h, r, t))
	elif p > 0.8:
		valid.append((h, r, t))
	else:
		train.append((h, r, t))
ent_n += 1; rel_n += 1
print(ent_n, rel_n)

triple = ''

# convert the training data
wtr = open("train.txt", "w")
dat = ''
for i in range(len(train)):
	tri = train[i]
	h, r, t = tri[0], tri[1], tri[2]
	dat += str(h) + '\t' + str(r) + '\t' + str(t) + '\n'
wtr.write(dat)
wtr.close()
triple += dat

# convert the validation data
wtr = open("valid.txt", "w")
dat = ''
for i in range(len(valid)):
	tri = valid[i]
	h, r, t = tri[0], tri[1], tri[2]
	dat += str(h) + '\t' + str(r) + '\t' + str(t) + '\n'
wtr.write(dat)
wtr.close()
triple += dat

# convert the testing data
wtr = open("test.txt", "w")
dat = ''
for i in range(len(test)):
	tri = test[i]
	h, r, t = tri[0], tri[1], tri[2]
	dat += str(h) + '\t' + str(r) + '\t' + str(t) + '\n'
wtr.write(dat)
wtr.close()
triple += dat

wtr = open("triple.txt", "w")
wtr.write(triple)
wtr.close()

wtr = open("entity2id.txt", "w")
dat = ''
for i in range(ent_n):
	dat += str(i) + '\t' + str(i) + '\n'
wtr.write(dat)
wtr.close()

wtr = open("relation2id.txt", "w")
dat = ''
for i in range(rel_n):
	dat += str(i) + '\t' + str(i) + '\n'
wtr.write(dat)
wtr.close()