import os
import torch
from torch import nn
from torch.utils import data
import numpy as np
import tqdm

# Rest of your code
dir_data_path = os.path.join(os.path.dirname(__file__), "./data")


def read_file(data_path, sp="\t"):
    result = {}
    with open(data_path) as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip().split(sp)
            result[row[0]] = int(row[1])
    return result

def open_train(data_path, sp="\t"):
    data_list = []
    with open(data_path) as f:
        lines = f.readlines()
        for line in lines:
            triple = line.strip().split(sp)
            if len(triple) != 3:
                continue
            data_list.append(tuple(triple))
    return data_list


class TransE(nn.Module):

    def __init__(self, entity_count, relation_count, norm=2, dim=50):
        super(TransE, self).__init__()
        self.norm = norm
        self.dim = dim
        self.entity_num = entity_count
        # self.entities_emb = self.emb_pretrain(entity_count)
        self.entities_emb = self._init_emb(entity_count)
        self.relations_emb = self._init_emb(relation_count)
        #offset=self._init_offset
    
    def _init_emb(self, num_embeddings):
        embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=self.dim)
        uniform_range = 6 / np.sqrt(self.dim)
        embedding.weight.data.uniform_(-uniform_range, uniform_range)
        embedding.weight.data = torch.div(embedding.weight.data,
                                          embedding.weight.data.norm(p=2, dim=1, keepdim=True))
        return embedding
    def emb_pretrain(self,num_embeddings):
        M = []
        dim=50
        ent_n=14541
        rel_n=237
        for k in range(dim):
            M.append((torch.rand(size = (1, rel_n)) - 0.5) * 2)


        raw = open("init_emb.txt", "r").readlines()
        ent_init_emb = torch.empty(size = (ent_n, dim))
        ent_id = 0
        for line in raw:
            s= line.strip().split()
            for i in range(len(s)):
                s[i] = float(s[i])
            s = torch.tensor(s)
		
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
        embedding_layer = nn.Embedding.from_pretrained(ent_init_emb)
        embedding_layer.weight.requires_grad = True
        return embedding_layer 
    def forward(self, positive_triplets: torch.LongTensor, negative_triplets: torch.LongTensor,offset):
        
        positive_distances=(self._distance(positive_triplets))
        negative_distances =(self._distance(negative_triplets))
        return positive_distances, negative_distances

    def _distance(self, triplets):
        heads = triplets[:, 0]
        relations = triplets[:, 1]

        tails = triplets[:, 2]
        # L1还是L2距离
       
        return nn.functional.mish((self.entities_emb(heads)+self.relations_emb(relations) - self.entities_emb(tails))).norm(p=self.norm, dim=1)
    def link_predict(self, head, relation, tail=None, k=10):
        # h_add_r: [batch size, embed size] -> [batch size, 1, embed size] -> [batch size, entity num, embed size]
        h_add_r = self.entities_emb(head) + self.relations_emb(relation)
        h_add_r = torch.unsqueeze(h_add_r, dim=1)
        h_add_r = h_add_r.expand(h_add_r.shape[0], self.entity_num, self.dim)
        # embed_tail: [batch size, embed size] -> [batch size, entity num, embed size]
        embed_tail = self.entities_emb.weight.data.expand(h_add_r.shape[0], self.entity_num, self.dim)
        # values: [batch size, k] scores, the smaller, the better
        # indices: [batch size, k] indices of entities ranked by scores
        values, indices = torch.topk(torch.norm(h_add_r - embed_tail, dim=2), k=self.entity_num, dim=1, largest=False)
        if tail is not None:
            tail = tail.view(-1, 1)
            rank_num = torch.eq(indices, tail).nonzero().permute(1, 0)[1]+1
            mrrank=torch.sum(rank_num)
            mrr = torch.sum(1/rank_num)
            hits_1_num = torch.sum(torch.eq(indices[:, :1], tail)).item()
            hits_3_num = torch.sum(torch.eq(indices[:, :3], tail)).item()
            hits_10_num = torch.sum(torch.eq(indices[:, :10], tail)).item()
            return mrrank,mrr, hits_1_num, hits_3_num, hits_10_num     # 返回一个batchsize, mrr的和，hit@k的和
        return indices[:, :k]
    
    def evaluate(self, data_loader):
        mrr_rank_sum=mrr_sum = hits_1_nums = hits_3_nums = hits_10_nums = 0
        num=0
        for heads, relations, tails in tqdm.tqdm(data_loader):
            num+=1
            mrr_rank_batch,mrr_sum_batch, hits_1_num, hits_3_num, hits_10_num = self.link_predict(heads.cuda(), relations.cuda(), tails.cuda())
            mrr_rank_sum+=mrr_rank_batch
            mrr_sum += mrr_sum_batch
            hits_1_nums += hits_1_num
            hits_3_nums += hits_3_num
            hits_10_nums += hits_10_num
        num=num*400
        return mrr_rank_sum/num,mrr_sum/num, hits_1_nums/num, hits_3_nums/num, hits_10_nums/num


class TripleDataset(data.Dataset):

    def __init__(self, entity2id, relation2id, triple_data_list):
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.data = triple_data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        head, relation, tail = self.data[index]
        head_id = self.entity2id[head]
        relation_id = self.relation2id[relation]
        tail_id = self.entity2id[tail]
        return head_id, relation_id, tail_id


if __name__ == '__main__':
    entity_id_dict = read_file(os.path.join(dir_data_path, 'entity2id.txt'))
    relation_id_dict = read_file(os.path.join(dir_data_path, 'relation2id.txt'))
    train_data = open_train(os.path.join(dir_data_path, 'train.txt'))
    valid_data = open_train(os.path.join(dir_data_path, 'valid.txt'))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Rest of your code
    batch_size = 400
    epochs = 200
    margin = 1.0
    train_dataset = TripleDataset(entity_id_dict, relation_id_dict, train_data)
    valid_dataset = TripleDataset(entity_id_dict, relation_id_dict, valid_data)
    train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_data_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Rest of your code

    model = TransE(len(entity_id_dict), len(relation_id_dict)).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
    criterion = nn.MarginRankingLoss(margin=margin, reduction='mean')

    print(f"Start training")
    alpha = torch.tensor(0.0, requires_grad=True)
    best_mrr = 0
    offset=0
    for epoch in range(epochs):
        model.train()
        all_loss = 0

        for i, (local_heads, local_relations, local_tails) in enumerate(train_data_loader):
            local_heads = local_heads.to(device)
            local_relations = local_relations.to(device)
            local_tails = local_tails.to(device)

            positive_triples = torch.stack((local_heads, local_relations, local_tails), dim=1).to(device)

            head_or_tail = torch.randint(high=2, size=local_heads.size()).to(device)
            random_entities = torch.randint(high=len(entity_id_dict), size=local_heads.size()).to(device)
            broken_heads = torch.where(head_or_tail == 1, random_entities, local_heads)
            broken_tails = torch.where(head_or_tail == 0, random_entities, local_tails)
            negative_triples = torch.stack((broken_heads, local_relations, broken_tails), dim=1).to(device)

            optimizer.zero_grad()
            pd, nd = model(positive_triples, negative_triples,alpha)

            loss = criterion(pd, nd, torch.tensor([-1], dtype=torch.long).to(device))
            loss.backward()
            all_loss += loss.item()
            optimizer.step()

            #if i % 1000 == 0:
                #print(f"Epoch: {epoch+1}/{epochs}, Batch: {i}, Average Loss: {all_loss / (i + 1)}")

        #print(f"Epoch: {epoch+1}/{epochs}, Total Loss: {all_loss}")

        if epoch % 10 == 9:
            print(f"Epoch: {epoch+1}/{epochs}, Total Loss: {all_loss}")
            mrr_raw,mrr, hits1, hits3, hits10 = model.evaluate(valid_data_loader)
            if mrr >= best_mrr:
                best_mrr = mrr
                improve = '*'
                torch.save(model.state_dict(), 'transE_best.pth')
            torch.save(model.state_dict(), 'transE_latest.pth')
            print(f'MRR(raw):{mrr_raw},[]MRR: {mrr}, Hit@1: {hits1}, Hit@3: {hits3}, Hit@10: {hits10}  {improve}')



