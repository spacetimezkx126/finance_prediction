import torch
from data import StockDataset
from torch.utils.data import DataLoader,random_split
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
from torch_geometric.nn import GCNConv,GATv2Conv
from sklearn.metrics import confusion_matrix
import io
def calculate_mcc(y_true, y_pred):
    """
    计算 MCC (Matthew's Correlation Coefficient) 指标。
    
    Args:
        y_true (list or np.ndarray): 真实标签。
        y_pred (list or np.ndarray): 预测标签。
    
    Returns:
        float: MCC 值。
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / denominator if denominator != 0 else 0.0
class ChannelAttention1D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(ChannelAttention1D, self).__init__()
        
        # MLP模块，处理hidden_dim维度的输入
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        
        # Sigmoid激活函数，将输出归一化到[0, 1]之间
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x的形状：[batch_size, hidden_dim]
        
        # 步骤1: 通过MLP计算通道注意力权重
        avg_out = F.relu(self.fc1(x))  # 输出大小: [batch_size, hidden_dim // reduction_ratio]
        avg_out = self.fc2(avg_out)    # 输出大小: [batch_size, hidden_dim]
        
        # 步骤2: 使用sigmoid生成通道注意力权重
        avg_out = self.sigmoid(avg_out)  # 输出大小: [batch_size, hidden_dim]
        
        # 将通道注意力权重应用到输入特征图上
        return x * avg_out  # 返回加权后的特征图

class TemporalGCN(nn.Module):
    """处理时序图数据的模块"""
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(TemporalGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATv2Conv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GATv2Conv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index):
        """
        Args:
            x: 节点特征 (num_nodes, input_dim)
            edge_index: 边的索引 (2, num_edges)
        """
        for conv in self.layers:
            x = F.relu(conv(x, edge_index))
        return x
class CrossAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(input_dim, hidden_dim)  # 投影到 query 空间
        self.key_proj = nn.Linear(input_dim, hidden_dim)    # 投影到 key 空间
        self.value_proj = nn.Linear(input_dim, hidden_dim)  # 投影到 value 空间
        self.query_proj1 = nn.Linear(input_dim, hidden_dim)  # 投影到 query 空间
        self.key_proj1 = nn.Linear(input_dim, hidden_dim)    # 投影到 key 空间
        self.value_proj1 = nn.Linear(input_dim, hidden_dim)  # 投影到 value 空间
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, features_a, features_b):
        """
        :param features_a: Tensor, shape (batch_size, seq_len_a, input_dim)
        :param features_b: Tensor, shape (batch_size, seq_len_b, input_dim)
        :return: updated_a, updated_b
        """
        # 投影到 query, key, value 空间
        query_a = self.query_proj(features_a)  # (batch_size, seq_len_a, hidden_dim)
        key_b = self.key_proj(features_b)      # (batch_size, seq_len_b, hidden_dim)
        value_b = self.value_proj(features_b)  # (batch_size, seq_len_b, hidden_dim)
        
        query_b = self.query_proj1(features_b)  # (batch_size, seq_len_b, hidden_dim)
        key_a = self.key_proj1(features_a)      # (batch_size, seq_len_a, hidden_dim)
        value_a = self.value_proj1(features_a)  # (batch_size, seq_len_a, hidden_dim)

        # 计算注意力分数
        attention_a_to_b = torch.matmul(query_a, key_b.transpose(-1, -2))  # (batch_size, seq_len_a, seq_len_b)
        attention_a_to_b = self.softmax(attention_a_to_b / (key_b.size(-1) ** 0.5))
        
        attention_b_to_a = torch.matmul(query_b, key_a.transpose(-1, -2))  # (batch_size, seq_len_b, seq_len_a)
        attention_b_to_a = self.softmax(attention_b_to_a / (key_a.size(-1) ** 0.5))

        # 计算加权值
        updated_a = torch.matmul(attention_b_to_a, value_a)  # (batch_size, seq_len_b, hidden_dim)
        updated_b = torch.matmul(attention_a_to_b, value_b)  # (batch_size, seq_len_a, hidden_dim)

        return updated_a, updated_b


class StockModelTGNN(nn.Module):
    def __init__(self, embed_dim, stock_feature_dim, hidden_dim, num_classes=2):
        super(StockModelTGNN, self).__init__()
        # 文本嵌入和 CNN 提取特征
        self.dropout = nn.Dropout(0.1)
        embedding = 'embedding_SougouNews.npz'
        self.embedding_pretrained = torch.tensor(
            np.load('/home/zhaokx/finance_prediction/embedding_SougouNews.npz')["embeddings"].astype('float32')) \
            if embedding != 'random' else None  # 预训练词向量
        embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300
        self.embedding = nn.Embedding.from_pretrained(self.embedding_pretrained, freeze=True)
        # self.embedding = nn.Embedding(186089, 300)
        # self.embedding = nn.Embedding(59042, 300)
        # 多核 CNN 提取文本特征
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 48, (k, embed)) for k in (2, 3, 4)]
        )
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(1, 48, (k, embed)) for k in (2, 3, 4)]
        )
        self.text_fc = nn.Linear(len((2, 3, 4)) * 48, 64)  # 将 CNN 输出降维到 hidden_dim
        self.text_fc1 =  nn.Linear(len((2, 3, 4)) * 48, 64)
        # 股票特征用时序图神经网络提取
        self.stock_tgnn = TemporalGCN(stock_feature_dim, hidden_dim, num_layers=5)
        # self.stock_fc = nn.Linear(hidden_dim, hidden_dim)
        self.stock_lstm = nn.LSTM(stock_feature_dim, hidden_dim, batch_first=True)
        self.stock_lstm1 = nn.LSTM(stock_feature_dim, hidden_dim, batch_first=True)
        self.stock_lstm2 = nn.LSTM(hidden_dim+hidden_dim, hidden_dim+hidden_dim, batch_first=True)
        self.gat2 =  GATv2Conv(hidden_dim, hidden_dim,edge_dim=30)
        self.gat22 = GATv2Conv(hidden_dim, hidden_dim,edge_dim=30)
        # 融合层
        self.fc =  nn.Linear(hidden_dim+hidden_dim, num_classes) # 两部分特征的融合输出
        self.window_size = 5
        self.edge_attr = nn.Linear(1,30)
        self.channel_attn = CrossAttention(hidden_dim, hidden_dim)
    def conv_and_pool(self, x, conv):
        """卷积 + 最大池化"""
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    def forward(self, texts, stock_features, edge_index,all_text,edge_index1,label):
        # 文本处理
        texts = texts.squeeze(0)  # 移除可能存在的多余维度
        edge_all = torch.cat([edge_index[:,:2,:0],edge_index1[:,:2,:]],dim=2)
        edge_type = torch.cat([edge_index[:,2,:0],edge_index1[:,2,:]],dim=1).float().permute(1,0)
        batch_size, num_docs, max_words = texts.size()

        texts = texts.view(-1, max_words)  # 将所有文档展平成单个批次
        embedded_texts = self.embedding(texts.to(device)).unsqueeze(1)  # 嵌入层处理 (batch*num_docs, 1, max_words, embed_dim)
        all_text1 = all_text.squeeze(0)
        embedded_texts1 = self.embedding(all_text1.to(device)).unsqueeze(1)
        # 使用多种卷积核处理文本并池化
        text_features = [
            self.conv_and_pool(embedded_texts, conv) for conv in self.convs
        ]

        text_features1 = [
            self.conv_and_pool(embedded_texts1, conv) for conv in self.convs1
        ]
        
        text_repr = torch.cat(text_features, dim=1)  # 合并不同卷积核的输出
        text_repr = text_repr.view(batch_size, num_docs, -1).mean(dim=1)  # 每个样本的文档取平均表示
        text_repr1 = torch.cat(text_features1, dim=1)  # 合并不同卷积核的输出
        # 股票特征处理（基于时序图神经网络）
        stock_features = stock_features.to(device)
        node_repr = self.stock_tgnn(stock_features[0], edge_index[0,:2,:])  # (num_nodes, hidden_dim)
        # # 将滑动窗口的数据输入到LSTM
        lstm_outputs = []
        # # 拼接所有窗口的输出
        # stock_output, _ = self.stock_lstm(stock_features)

        #       # 前几个时间步（填充部分）
        # for i in range(self.window_size - 1):
        #     # 填充：直接将原始特征加入lst_outputs，填充的数量是window_size-1
        #     lstm_outputs.append(stock_output[:, i, :])  # 每个时间步的特征，保留维度(batch, 1, hidden_dim)
        
        # # 对剩余时间步（从window_size到最后）进行LSTM计算
        # for i in range(self.window_size - 1, stock_features.size(1)):
        #     window = stock_features[:, i - self.window_size + 1:i + 1, :]  # 获取当前时间步及其前window_size-1个时间步
        #     # print(window,stock_features[:,i,:],label[0,i])
        #     stock_output, _ = self.stock_lstm1(window)  # 将滑动窗口传入LSTM
        #     lstm_outputs.append(stock_output[:, -1, :])  # 只取每个窗口的最后一个时间步输出
        
        # # 将所有时间步的输出拼接在一起
        # stock_output1 = torch.stack(lstm_outputs, dim=0)

        node_cat = torch.cat([torch.ones(node_repr.shape[0],node_repr.shape[1]).to(device),text_repr1],dim=0)
        edge_type = self.edge_attr(edge_type)
        node_gat = self.gat2(node_cat,edge_all[0],edge_attr=edge_type)
        node_stock = node_gat[:node_repr.shape[0]]
        combined_repr = torch.cat([stock_features[0]],dim=1)
        # 拼接所有窗口的输出
        stock_output, _ = self.stock_lstm(combined_repr)
        # 前几个时间步（填充部分）
        for i in range(self.window_size - 1):
            # 填充：直接将原始特征加入lst_outputs，填充的数量是window_size-1
            lstm_outputs.append(torch.cat([stock_output[i, :].unsqueeze(0),node_stock[i,:].unsqueeze(0)],dim=1))  # 每个时间步的特征，保留维度(batch, 1, hidden_dim)
        
        # 对剩余时间步（从window_size到最后）进行LSTM计算
        for i in range(self.window_size - 1, stock_features.size(1)):
            window1 = stock_features[0][i - self.window_size + 1:i + 1, :]  # 获取当前时间步及其前window_size-1个时间步
            window2 = node_stock[i - self.window_size + 1:i + 1, :]

            window1, _ = self.stock_lstm1(window1)  # 将滑动窗口传入LSTM
            updated_a,updated_b = self.channel_attn(window1,window2)
            updated_c = torch.cat([updated_a,updated_b],dim=1)
            updated_c,_ = self.stock_lstm2(updated_c)
            lstm_outputs.append(updated_c[-1,:].unsqueeze(0))  # 只取每个窗口的最后一个时间步输出
        
        # 将所有时间步的输出拼接在一起
        stock_output1 = torch.stack(lstm_outputs, dim=0)

        # 二分类输出
        output = self.fc(stock_output1)  # (batch_size, num_classes)
        output = output.squeeze(1)
        return output

# class StockModel(nn.Module):
#     def __init__(self, embed_dim, stock_feature_dim, hidden_dim, num_classes=2):
#         super(StockModel, self).__init__()
#         # 文本嵌入和 CNN 提取特征
#         embedding = 'embedding_SougouNews.npz'
#         self.embedding_pretrained = torch.tensor(
#             np.load('/home/zhaokx/legal_case_retrieval/law_case_retrieval_temp1/others/embedding_SougouNews.npz')["embeddings"].astype('float32')) \
#             if embedding != 'random' else None  # 预训练词向量
#         embed = self.embedding_pretrained.size(1) \
#             if self.embedding_pretrained is not None else 300
#         self.embedding = nn.Embedding.from_pretrained(self.embedding_pretrained, freeze=True)
#         # print(self.embedding)
#         # 多核 CNN 提取文本特征
#         self.convs = nn.ModuleList(
#             [nn.Conv2d(1, 64, (k, embed)) for k in (2, 3, 4)]
#         )
#         self.text_fc = nn.Linear(len((2, 3, 4)) * 64, hidden_dim)  # 将 CNN 输出降维到 hidden_dim

#         # 股票特征 LSTM 提取时序特征
#         self.stock_lstm = nn.LSTM(stock_feature_dim, hidden_dim, batch_first=True, bidirectional=True)
#         self.stock_fc = nn.Linear(hidden_dim * 2, hidden_dim)  # 双向 LSTM 输出降维

#         # 融合层
#         self.fc = nn.Linear(192, num_classes)  # 两部分特征的融合输出

#     def conv_and_pool(self, x, conv):
#         """卷积 + 最大池化"""
#         x = F.relu(conv(x)).squeeze(3)
#         x = F.max_pool1d(x, x.size(2)).squeeze(2)
#         return x

#     def forward(self, texts, stock_features):
#         # 文本处理
#         texts = texts.squeeze(0)
#         # print(texts.size(),"43**")
#         batch_size, num_docs, max_words = texts.size()
#         texts = texts.view(-1, max_words)  # 展平每个文档
#         # print(texts.shape)
#         # print(torch.max(texts))
#         xtext = self.embedding(texts.to(device)).unsqueeze(1)  # (batch*num_docs, 1, max_words, embed_dim)
#         text_repr = torch.cat([self.conv_and_pool(xtext, conv) for conv in self.convs], 1)
#         text_repr = self.text_fc(text_repr)  # (batch*num_docs, hidden_dim)
#         text_repr = text_repr.view(batch_size, num_docs, -1).mean(dim=1)  # 每个样本的文档取平均

#         # 股票特征处理
#         stock_features = stock_features.to(device)  # (batch_size, seq_len, stock_feature_dim)
#         # print(stock_features.shape)
#         stock_output, _ = self.stock_lstm(stock_features)  # (batch_size, seq_len, hidden_dim*2)
#         # print(stock_output.shape)
#         # stock_repr = self.stock_fc(stock_output[:, -1, :])  # 取最后一个时间步的特征
#         # print(stock_output.shape)
#         # 融合
#         # print(text_repr.shape,stock_output.shape)
#         combined = torch.cat((text_repr, stock_output.squeeze(0)), dim=1)  # (batch_size, hidden_dim*2)

#         # 二分类
#         output = self.fc(combined)  # (batch_size, num_classes)
#         return output

# 主函数

def custom_collate_fn(batch):
    # 过滤掉返回 None 的样本
    batch = [item for item in batch if item is not None]

    # print("74**",batch[0])
    if len(batch[0]['time']) ==0:
        return None
    # 如果过滤后为空，抛出异常或返回空批次
    if len(batch) == 0:
        return None

    # 正常处理有效样本
    times = [item["time"] for item in batch]
    texts = [item["texts"] for item in batch]
    stock_features = [item["stock_features"] for item in batch]
    labels = [item["label"] for item in batch]
    edge_index = [item["edge_index"] for item in batch]
    all_text = [item["all_text"] for item in batch]
    edge_index1 = [item["edge_index1"] for item in batch]
    view_text = [item["view_text"] for item in batch]
    # print("86",texts,torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0))
    return {
        "time": times,
        "texts": torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0),
        "stock_features": torch.nn.utils.rnn.pad_sequence(stock_features, batch_first=True, padding_value=0),
        "label": torch.stack(labels),
        "edge_index" :torch.stack(edge_index),
        "edge_index1" : torch.stack(edge_index1),
        "all_text" : torch.stack(all_text),
        "view_text": view_text
    }

# # 使用自定义 collate_fn
# data_loader = DataLoader(dataset, batch_size=32, collate_fn=custom_collate_fn)
def main():
    # 数据集参数
    # text_dir = "/path/to/text/data"
    # sequence_dir = "/path/to/sequence/data"
    # vocab = {"<PAD>": 0, "<UNK>": 1, "word1": 2, "word2": 3, ...}  # 示例词汇表
    # max_seq_len = 100
    dataset_train = StockDataset("/home/zhaokx/finance_prediction/CMIN-CN/news/preprocessed","/home/zhaokx/finance_prediction/CMIN-CN/price/preprocessed","/home/zhaokx/finance_prediction/dict.pkl",mode='train')
    dataset_val = StockDataset("/home/zhaokx/finance_prediction/CMIN-CN/news/preprocessed","/home/zhaokx/finance_prediction/CMIN-CN/price/preprocessed","/home/zhaokx/finance_prediction/dict.pkl",mode='val')
    batch_size = 1
    num_epochs = 50
    # 创建数据加载器
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # 模型参数
    embed_dim = 128
    stock_feature_dim = 3
    hidden_dim = 144

    # 模型、损失函数和优化器
    model = StockModelTGNN(embed_dim, stock_feature_dim, hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 训练循环
    # num_epochs = 10  # 假设您有 10 个训练轮次
    accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_dataloader:
            if batch is None:
                continue
            texts = batch["texts"]  # (batch_size, num_docs, max_words)
            stock_features = batch["stock_features"]  # (batch_size, num_docs, feature_dim)
            labels = batch["label"]  # (batch_size, )
            edge_index = batch["edge_index"]
            edge_index1 = batch["edge_index1"]
            all_text = batch["all_text"]
            view_text = batch["view_text"]

            # 前向传播
            outputs = model(texts.to(device), stock_features.to(device),edge_index.to(device),all_text.to(device),edge_index1.to(device),labels.to(device))  # (batch_size, 2)

            # 计算损失
            # print(outputs.shape, labels.squeeze(0).shape)
            loss = criterion(outputs, labels.squeeze(0).to(device))
            print(loss)
            # print(len(view_text),len(outputs),len(labels))

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels.to(device)).sum().item()
            total += labels.size(1)
            if accuracy>80:
                for i in range(len(outputs)):
                    print("train:",view_text[0][i],outputs[i],labels[0][i])

        accuracy = correct / total * 100
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # 验证集评估
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_accuracy = 0
        pred_all = []
        label_all = []
        with torch.no_grad():
            for batch in val_dataloader:
                if batch is None:
                    continue
                texts = batch["texts"]
                stock_features = batch["stock_features"]
                labels = batch["label"]
                edge_index = batch["edge_index"]
                all_text = batch["all_text"]
                edge_index1 = batch["edge_index1"]
                view_text = batch["view_text"]
                outputs = model(texts.to(device), stock_features.to(device),edge_index.to(device),all_text.to(device),edge_index1.to(device),labels.to(device))
                loss = criterion(outputs, labels.squeeze(0).to(device))
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels.to(device)).sum().item()
                # print(predicted,labels)
                val_total += labels.size(1)
                pred_all.extend(predicted.tolist())
                label_all.extend(labels[0].tolist())
                if accuracy>80:
                    for i in range(len(outputs)):
                        print("val:",view_text[0][i],outputs[i],labels[0][i])
        print(label_all,pred_all)
        mcc = calculate_mcc(label_all,pred_all)
        val_accuracy = val_correct / val_total * 100
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, Validation Mcc: {mcc:.2f}%")
    # # 加载数据集
    # dataset = StockDataset("/home/zhaokx/CMIN-Dataset-main/CMIN-CN/news/preprocessed","/home/zhaokx/CMIN-Dataset-main/CMIN-CN/price/preprocessed","/home/zhaokx/CMIN-Dataset-main/dict.pkl")
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # # 模型参数
    # # vocab_size = len(vocab)
    # embed_dim = 128
    # stock_feature_dim = 6
    # hidden_dim = 64

    # # 模型、损失函数和优化器
    # model = StockModel(embed_dim, stock_feature_dim, hidden_dim).to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # # 训练循环
    # for epoch in range(num_epochs):
    #     model.train()
    #     epoch_loss = 0.0
    #     correct = 0
    #     total = 0

    #     for batch in dataloader:
            
    #         texts = batch["texts"]  # (batch_size, num_docs, max_words)
    #         stock_features = batch["stock_features"]  # (batch_size, num_docs, feature_dim)
    #         labels = batch["label"]  # (batch_size, )

    #         # 前向传播
    #         outputs = model(texts.to(device), stock_features.to(device))  # (batch_size, 2)
    #         # print(outputs.shape,labels.shape)
    #         # print(outputs,labels)
    #         loss = criterion(outputs, labels.squeeze(0).to(device))
    #         print(loss)
    #         # 反向传播与优化
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         epoch_loss += loss.item()

    #         # 计算准确率
    #         _, predicted = torch.max(outputs, 1)
    #         correct += (predicted == labels.to(device)).sum().item()
    #         total += labels.size(1)
    #         # print((predicted == labels.to(device)).sum().item(),labels.size(1))
    #     accuracy = correct / total * 100
    #     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")


    # print("训练完成！")

if __name__ == "__main__":
    main()
