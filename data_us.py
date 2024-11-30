import torch
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
from datetime import datetime, timedelta
import pandas as pd

# # 文本数据加载和处理
# def load_text_data(text_file):
#     with open(text_file, 'r', encoding='utf-8') as f:
#         text = f.read()
#     return text.strip().split()  # 按空格或其他分隔符分词

# class StockDataset(Dataset):
#     def __init__(self, text_dir, sequence_dir, word_to_idx, max_seq_len=100):
#         """
#         初始化数据集。
#         Args:
#             text_dir (str): 文本数据目录。
#             sequence_dir (str): 序列数据目录。
#             word_to_idx (dict): 单词到索引的映射字典。
#             max_seq_len (int): 序列的最大长度（用于截断或填充）。
#         """
#         # self.data = self._group_by_time(data)
#         self.text_dir = text_dir
#         self.folder = os.listdir(self.text_dir)
#         self.sequence_dir = sequence_dir
#         self.vocab = pkl.load(open("/home/zhaokx/KELLER/dict.pkl", 'rb'))
#         for comp in self.folder:
#             file_list = os.listdir(os.path.join(self.text_dir,comp))



#     def _group_by_time(self, data):
#         """
#         Group data by 'created_at' timestamp.
#         """
#         grouped = {}
#         for entry in data:
#             text = entry["text"]
#             time = entry["created_at"]
#             if time not in grouped:
#                 grouped[time] = []
#             grouped[time].append(text)
#         return list(grouped.values())

#     def _tokenize_and_pad_text(self, text):
#         """
#         Tokenize and pad a single text entry to `max_words`.
#         """
#         indices = [self.vocab.get(char, self.vocab["<UNK>"]) for char in text]
#         padded = indices[:self.max_words] + [self.vocab["<PAD>"]] * max(0, self.max_words - len(indices))
#         return padded

#     def _pad_time_step(self, texts):
#         """
#         Pad/truncate texts to `max_docs` and each text to `max_words`.
#         """
#         padded_texts = [self._tokenize_and_pad_text(text) for text in texts]
#         # Add zero-padded texts if fewer than `max_docs`
#         empty_text = [self.vocab["<PAD>"]] * self.max_words
#         while len(padded_texts) < self.max_docs:
#             padded_texts.append(empty_text)
#         return padded_texts[:self.max_docs]

#     def __getitem__(self, idx):
#         """
#         Get a single time step as a tensor of shape `(max_docs, max_words)`.
#         """
#         texts = self.data[idx]
#         padded_time_step = self._pad_time_step(texts)
#         return torch.tensor(padded_time_step, dtype=torch.long)

#     def __len__(self):
#         """
#         Return the number of time steps.
#         """
#         return len(self.data)

import os
import json
import torch
from torch.utils.data import Dataset
import pickle as pkl


class StockDataset(Dataset):
    def __init__(self, text_dir, sequence_dir, max_seq_len=40,mode='train'):
        """
        初始化数据集。
        Args:
            text_dir (str): 文本数据目录。
            sequence_file (str): 股票序列文件路径。
            vocab_file (str): 词汇表文件路径。
            max_seq_len (int): 文本序列的最大长度（用于截断或填充）。
        """
        self.text_dir = text_dir
        self.sequence_dir = sequence_dir
        # self.vocab = self.load_glove_embeddings()
        # self.vocab = pkl.load(open("/home/zhaokx/CMIN-Dataset-main/dict_tweet.pkl", 'rb'))
        self.vocab = pkl.load(open("/home/zhaokx/CMIN-Dataset-main/cmin-us.pkl", 'rb'))  # 加载词汇表
        self.max_seq_len = max_seq_len
        self.mode = mode
        # 加载文本数据
        self.text_data = self._load_text_data()
        # 加载股票序列数据
        self.stock_data = self._load_stock_data()
        # 对齐数据
        self.train_period = ['2018-01-01','2021-04-30']
        self.dev_period = ['2021-09-01','2021-12-31']
        # self.train_period = ['2014-01-01','2014-12-31']
        # self.dev_period = ['2015-10-01','2015-12-31']
        self.data = self._align_data_by_time()

    def _load_text_data(self):
        """
        加载文本数据并按时间戳组织。
        """
        text_data = {}
        for comp in os.listdir(self.text_dir):
            comp_path = os.path.join(self.text_dir, comp)
            if os.path.isdir(comp_path):
                text_data[comp] ={}
                file_list = os.listdir(comp_path)
                file_list.sort()
                for file_name in file_list:
                    file_path = os.path.join(comp_path, file_name)
                    # print(file_path,"124**")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            entry = json.loads(line.strip())
                            time = entry["created_at"].split(" ")[0]
                            # time = entry["created_at"]
                            # parsed_time = datetime.strptime(time, "%a %b %d %H:%M:%S +0000 %Y")

                            # # #转换为目标格式
                            # time = parsed_time.strftime("%Y-%m-%d")
                            text = entry["text"]
                            if time not in text_data[comp]:
                                text_data[comp][time] = []
                            text_data[comp][time].append(text)
            # break
        # print(text_data)
        return text_data
    def load_glove_embeddings(self, file_path="/home/zhaokx/CMIN-Dataset-main/glove.twitter.27B.200d.txt", vocab_size=50000):
        vocab = {}
        with open(file_path, encoding="utf8") as f:
            for idx, line in enumerate(f):
                if idx >= vocab_size-1:
                    break
                word = line.split()[0]
                vocab[word] = idx  # word -> index
        vocab["<unk>"] = len(vocab)-2
        vocab["<pad>"] = len(vocab)-1
        return vocab

    def _load_stock_data(self):
        """
        加载股票序列数据。
        """
        stock_data = {}
        for sequence_file in os.listdir(self.sequence_dir):
            path = os.path.join(self.sequence_dir,sequence_file)
            stock_data[sequence_file.replace(".txt","")] ={}
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    time = parts[0]
                    features = [float(x) for x in parts[2:5]]
                    stock_data[sequence_file.replace(".txt","")][time] = features
            # break
        return stock_data

    def _align_data_by_time(self):
        """
        对齐文本数据和股票数据。
        """
        aligned_data_all = []
        # print(self.stock_data)
        for comp, time_texts in self.text_data.items():
            # print(comp)
            aligned_data = []
            # print(comp in self.stock_data)
            if comp in self.stock_data:
                for time in time_texts:
                    # print(time,time in self.stock_data[comp])
                    if time in self.stock_data[comp]:
                        stock_features = self.stock_data[comp][time]
                        # print(stock_features,"180**")
                        if self.mode == 'train':
                            test = datetime.strptime(time, '%Y-%m-%d').date()
                            # print("183***",time,time_texts[time],stock_features)
                            if datetime.strptime(self.train_period[0], '%Y-%m-%d').date()< test < datetime.strptime(self.train_period[1], '%Y-%m-%d').date():
                                aligned_data.append({
                                    "time": time,
                                    "texts": time_texts[time],
                                    "stock_features": stock_features
                                })
                        if self.mode == 'val':
                            test = datetime.strptime(time, '%Y-%m-%d').date()
                            if datetime.strptime(self.dev_period[0], '%Y-%m-%d').date()< test < datetime.strptime(self.dev_period[1], '%Y-%m-%d').date():
                                aligned_data.append({
                                    "time": time,
                                    "texts": time_texts[time],
                                    "stock_features": stock_features
                                })
                        
                        
                aligned_data_all.append(aligned_data)
                # break
        # print(aligned_data)
        return aligned_data_all
    def _pad_texts_to_equal_length(self, tokenized_texts, max_docs=10):
        """
        对 tokenized_texts 填充或截断，使其数量达到 max_docs。
        Args:
            tokenized_texts (list of list): 当前样本中的所有文本的索引化序列。
            max_docs (int): 统一的文本序列数量。
        Returns:
            padded_texts (list of list): 填充后的 tokenized_texts。
        """
        empty_text = [self.vocab["<pad>"]] * self.max_seq_len # 单个空文本的填充值
        # 如果数量不足，填充空文本；如果超出，进行截断
        if len(tokenized_texts) < max_docs:
            tokenized_texts.extend([empty_text] * (max_docs - len(tokenized_texts)))
        else:
            tokenized_texts = tokenized_texts[:max_docs]
        return tokenized_texts

    def _tokenize_and_pad_text(self, text):
        """
        将文本索引化并填充到固定长度。
        """
        # print("236",text,self.vocab)
        indices = [self.vocab.get(char, self.vocab["<unk>"]) for char in text]
        # print(indices,self.max_seq_len)
        padded = indices[:self.max_seq_len] + [self.vocab["<pad>"]] * max(0, self.max_seq_len - len(indices))
        # print(padded)
        return padded

    def construct_edge_index(self,samples,mappings):
        """
        构造 edge_index，根据样本中时间戳的文本和股票数据。
        Args:
            samples (list of dict): 当前样本数据。
        Returns:
            edge_index (torch.LongTensor): 图的边索引 (2, num_edges)。
        """
        edges = []
        edges1 = []
        num_nodes = len(samples)  # 节点总数（时间步数量）
        # print("250",num_nodes)
        # 时间相关边：连接相邻时间戳的节点
        for i in range(num_nodes - 3):
            edges.append((i, i + 1,0))
            # edges.append((i, i + 2,1))
            # edges.append((i, i + 3,2))
            # edges.append((i, i + 4))
            # edges.append((i, i + 5))

            # edges.append((i + 1, i))
        for key in mappings:
            edges1.append((key,mappings[key],3))
        if len(mappings)!=0:
            keys = mappings.keys()
            min_key = min(keys)
            max_key = max(keys)
            for i in range(min_key,max_key):
                edges1.append((i,i+1,4))
        # # 跨类型边：连接文本节点与股票特征节点
        # for i in range(num_nodes):
        #     edges.append((i, i))  # 自环边
            # 可扩展为其他跨节点连接规则

        # 转换为 edge_index 格式
        if len(edges)!=0:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            if len(edges1)!=0:
                edge_index1 = torch.tensor(edges1, dtype=torch.long).t()
                return edge_index,edge_index1
            return edge_index,None
        return None,None

    def __getitem__(self, idx):
        """
        获取单个样本，包括文本和股票特征。
        """
        samples = self.data[idx]
        all_text_tensors = []
        all_stock_tensors = []
        times = []
        labels = []
        mappings = {}
        text_id = len(samples)
        # print(samples)
        all_texts = []
        temp_id = text_id
        for i, sample in enumerate(samples):
            
            texts = sample["texts"]
            stock_features = sample["stock_features"]
            time = sample['time']
            # 对所有文本进行索引化和填充
            # print("??299",samples[i])
            tokenized_texts = [self._tokenize_and_pad_text(text) for text in texts]
            # print("301**",texts)
            for k in range(len(tokenized_texts)):
                mappings[text_id] = i
                # print(text_id-len(samples),len(texts),i,len(samples))
                # print(texts[text_id-temp_id],samples[i])
                text_id += 1
            temp_id = text_id
            for text in texts:
                all_texts.append(torch.tensor(self._tokenize_and_pad_text(text),dtype=torch.long))
            
            close_today = samples[i]["stock_features"][2]
            close_tomorrow =  samples[i+1]["stock_features"][2] if i+1<len(samples) else close_today

            label = 1 if close_tomorrow > close_today else 0
            # print(label,close_today,close_yesterday)
            # 将文本和股票特征转为张量
            tokenized_texts = self._pad_texts_to_equal_length(tokenized_texts)
            text_tensor = torch.tensor(tokenized_texts, dtype=torch.long)
            # print(text_tensor)
            # if text_tensor is not None:
            #     print(text_tensor.shape)
            # print(texts,stock_features,time,text_tensor)
            stock_tensor = torch.tensor(stock_features, dtype=torch.float)
            all_text_tensors.append(text_tensor)
            all_stock_tensors.append(stock_tensor)
            times.append(time)
            labels.append(torch.tensor(label))
            # print(stock_features,label)
            # print("318**",stock_tensor,text_tensor,label)
        # print(torch.stack(all_text_tensors).shape)
        edge_index,edge_index1 = self.construct_edge_index(samples,mappings)
        return {
        "time": times,  # 时间戳信息，列表形式
        "texts":  torch.stack(all_text_tensors) if all_text_tensors else None,  # 文本张量 (num_samples, num_docs, max_words)
        "stock_features": torch.stack(all_stock_tensors) if all_stock_tensors else None,  # 股票特征张量 (num_samples, num_docs, feature_dim)
        "label" : torch.stack(labels) if labels else None,
        "edge_index": edge_index,
        "edge_index1" : edge_index1,
        "all_text" : torch.stack(all_texts) if all_texts else None,
    }

    def __len__(self):
        """
        返回数据的长度。
        """
        return len(self.data)
if __name__ == "__main__":
    data = StockDataset("/home/zhaokx/CMIN-Dataset-main/CMIN-US/news/preprocessed","/home/zhaokx/CMIN-Dataset-main/CMIN-US/price/preprocessed")
    data[0]

