import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 读取训练文件
train_file_path = 'GraphAutoencoderoutput_Train_data2.0.xlsx'  # 替换为训练文件的实际路径
train_df = pd.read_excel(train_file_path)

# 读取测试文件
test_file_path = 'GraphAutoencoderoutput_TEST_data2.0.xlsx'  # 替换为测试文件的实际路径
test_df = pd.read_excel(test_file_path)

# 特征列名
feature_columns = [
    'Bid or Low Price', 'Ask or High Price', 'Price or Bid/Ask Average', 'Volume', 'Returns',
    'Bid', 'Ask', 'Shares Outstanding', 'Cumulative Factor to Adjust Prices',
    'Cumulative Factor to Adjust Shares/Vol', 'Open Price', 'NASDAQ Number of Trades',
    'Returns without Dividends', 'Value-Weighted Return-incl. dividends',
    'Value-Weighted Return-excl. dividends', 'Equal-Weighted Return-incl. dividends',
    'Equal-Weighted Return-excl. dividends', 'Return on the S&P 500 Index',
    'Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5', 'Feature_6',
    'Feature_7', 'Feature_8', 'Feature_9', 'Feature_10', 'Feature_11', 'Feature_12',
    'Feature_13', 'Feature_14', 'Feature_15', 'Feature_16', 'Feature_17', 'Feature_18',
    'Feature_19', 'Feature_20', 'Feature_21', 'Feature_22', 'Feature_23', 'Feature_24',
    'Feature_25', 'Feature_26', 'Feature_27', 'Feature_28', 'Feature_29', 'Feature_30',
    'Feature_31', 'Feature_32'
    ]

# feature_columns = [
#     'Bid or Low Price', 'Ask or High Price', 'Price or Bid/Ask Average', 'Volume', 'Returns',
#     'Bid', 'Ask', 'Shares Outstanding', 'Cumulative Factor to Adjust Prices',
#     'Cumulative Factor to Adjust Shares/Vol', 'Open Price', 'NASDAQ Number of Trades',
#     'Returns without Dividends', 'Value-Weighted Return-incl. dividends',
#     'Value-Weighted Return-excl. dividends', 'Equal-Weighted Return-incl. dividends',
#     'Equal-Weighted Return-excl. dividends', 'Return on the S&P 500 Index'
#     ]



# 处理缺失值，使用前向填充
train_df[feature_columns] = train_df[feature_columns].fillna(method='ffill')

# 按 PERMNO 分组处理训练集数据
grouped_train = train_df.groupby('PERMNO')

# 设置时间窗口大小为5
window_size = 5

# 组织训练集数据为序列样本
X_train_sequence = []
y_train_sequence = []
for permno, group in grouped_train:
    group = group.sort_values(by='Names Date')
    X = group[feature_columns].values
    y = group['Class'].values
    
    # 标准化特征数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for i in range(len(X_scaled) - window_size):
        X_train_sequence.append(X_scaled[i:i+window_size])
        y_train_sequence.append(y[i+window_size])

X_train_sequence = np.array(X_train_sequence)
y_train_sequence = np.array(y_train_sequence) + 1  # 将标签值从 [-1, 0, 1] 映射到 [0, 1, 2]

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train_sequence, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_sequence, dtype=torch.long)

# 创建训练集数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义LSTM+MLP模型
class LSTMMLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, mlp_dim, num_classes):
        super(LSTMMLPModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(0.5),  # 添加 Dropout 正则化
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步的隐藏状态
        return self.mlp(lstm_out)

# 初始化模型、损失函数和优化器
input_dim = len(feature_columns)
hidden_dim = 128
mlp_dim = 64
num_classes = 3  # 上涨/持平/下跌
model = LSTMMLPModel(input_dim, hidden_dim, mlp_dim, num_classes)

# 模型权重初始化
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

model.apply(init_weights)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
num_epochs = 30
for epoch in range(num_epochs):
    model.train() # 启用模型中的特定训练层
    train_loss = 0.0
    all_preds = []
    all_labels = []
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
        
        # 记录预测结果和真实标签
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(y_batch.numpy())
    
    train_loss /= len(train_loader.dataset)
    
    # 将标签值从 [0, 1, 2] 映射回 [-1, 0, 1]
    all_labels = np.array(all_labels) - 1
    all_preds = np.array(all_preds) - 1
    
    train_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}')

# 处理测试集数据
test_df[feature_columns] = test_df[feature_columns].fillna(method='ffill')

# 按 PERMNO 分组处理测试集数据
grouped_test = test_df.groupby('PERMNO')

# 组织测试集数据为序列样本
X_test_sequence = []
y_test_sequence = []
for permno, group in grouped_test:
    group = group.sort_values(by='Names Date')
    X = group[feature_columns].values
    y = group['Class'].values
    
    # 标准化特征数据 这里的标准化方式存在问题，不能在测试集上进行fit
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for i in range(len(X_scaled) - window_size):
        X_test_sequence.append(X_scaled[i:i+window_size])
        y_test_sequence.append(y[i+window_size])

X_test_sequence = np.array(X_test_sequence)
y_test_sequence = np.array(y_test_sequence) + 1  # 将标签值从 [-1, 0, 1] 映射到 [0, 1, 2]

# 转换为PyTorch张量
X_test_tensor = torch.tensor(X_test_sequence, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_sequence, dtype=torch.long)

# 创建测试集数据加载器
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 模型评估
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)
        print(preds)
        all_preds.extend(preds.numpy())
        all_labels.extend(y_batch.numpy())

# 将标签值从 [0, 1, 2] 映射回 [-1, 0, 1]
all_labels = np.array(all_labels) - 1
all_preds = np.array(all_preds) - 1

# 计算测试集的 F1 分数
test_f1 = f1_score(all_labels, all_preds, average='weighted')
print(f'Test F1 Score: {test_f1:.4f}')

# 将预测结果添加到测试数据框
test_df['Predicted_Class'] = np.nan  # 添加一列用于存储预测结果

# 将预测结果填入数据框
current_row = 0
for permno, group in grouped_test:
    group = group.sort_values(by='Names Date')
    for i in range(len(group) - window_size):
        idx = group.index[i + window_size]
        test_df.loc[idx, 'Predicted_Class'] = all_preds[current_row]  # 使用原始标签范围
        current_row += 1

# 保存包含预测结果的测试数据框到新的 Excel 文件
output_test_path = 'GraphAutoencoderoutput_Test_data_with_predictions.xlsx'
test_df.to_excel(output_test_path, index=False)

print(f"Predictions saved to {output_test_path}")