import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import time

# 参数配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 24  # 用过去24小时预测未来
BATCH_SIZE = 128
num_epochs = 10

# 数据加载与预处理（仅划分训练集和测试集）
def load_data(csv_path, seq_len=SEQ_LEN, split_ratio=(0.8, 0.2)):
    # 读取数据并提取OT列
    df = pd.read_csv(csv_path, parse_dates=['date'])
    data = df[['OT']].values  # (n_samples, 1)
    
    # 标准化
    scaler = StandardScaler()
    train_split = int(len(data) * split_ratio[0])
    
    # 仅用训练数据拟合scaler
    scaler.fit(data[:train_split])
    data_scaled = scaler.transform(data)
    
    # 创建滑动窗口数据集
    X, y = [], []
    for i in range(len(data_scaled)-seq_len):
        X.append(data_scaled[i:i+seq_len])
        y.append(data_scaled[i+seq_len])
    X = np.array(X)
    y = np.array(y)
    
    # 划分数据集（保持时间顺序）
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]
    
    # 转为Tensor
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    
    return (X_train, y_train), (X_test, y_test), scaler

# 创建数据加载器
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {'input': self.X[idx], 'target': self.y[idx]}

# LSTM 模型定义
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=512, num_layers=1, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # 初始化隐藏状态和单元状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        
        # 通过全连接层
        out = self.fc(out)
        
        return out

# MinLSTM 模型定义
class MinLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, device=None, dtype=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, hidden_size * 3, bias=False, device=device, dtype=dtype)
        self.output_layer = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)  # Map hidden state to output

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        f, i, h = torch.chunk(self.linear(x), chunks=3, dim=-1)
        diff = F.softplus(-f) - F.softplus(-i)
        log_f = -F.softplus(diff)
        log_i = -F.softplus(-diff)
        log_h_0 = self.log_g(h_prev)
        log_tilde_h = self.log_g(h)
        log_coeff = log_f.unsqueeze(1)
        log_val = torch.cat([log_h_0.unsqueeze(1), (log_i + log_tilde_h)], dim=1)
        h_t = self.parallel_scan_log(log_coeff, log_val)
        # Use the last hidden state for prediction
        output = self.output_layer(h_t[:, -1, :])
        return output

    def parallel_scan_log(self, log_coeffs, log_values):
        a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0)).squeeze(1)
        log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1).squeeze(1)
        log_h = a_star + log_h0_plus_b_star
        return torch.exp(log_h)  # Returns [batch, seq + 1, chn]

    def g(self, x):
        return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))

    def log_g(self, x):
        return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

# 修改训练函数（移除验证相关逻辑）
def train_and_evaluate(model, train_dataloader, test_dataloader,
                      loss_fn, optimizer, num_epochs=10):
    # 初始化统计量
    start_time = time.time()
    train_losses = []
    memory_usage = []
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        training_loss = 0.0
        for idx, batch in enumerate(train_dataloader):
            # 重置CUDA内存统计
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            inputs = batch['input']
            targets = batch['target']
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 记录内存使用（峰值）
            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated()
                memory_usage.append(peak_mem)
            
            # 记录损失
            training_loss += loss.item()
            train_losses.append(loss.item())
            
            if idx % 100 == 0:
                print(f'Epoch: {epoch}, Step: {idx}, Train Loss: {training_loss/(idx+1):.4f}')

        avg_train_loss = training_loss/len(train_dataloader)
        print(f'Epoch: {epoch} => Avg Train Loss: {avg_train_loss:.4f}')
    
    # 计算总时间和平均内存
    total_time = time.time() - start_time
    avg_memory = sum(memory_usage)/len(memory_usage)/(1024**2) if memory_usage else 0  # 转换为MB
    
    # 测试评估
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for batch in test_dataloader:
            inputs = batch['input']
            targets = batch['target']
            outputs = model(inputs)
            y_pred.extend(outputs.cpu().numpy())
            y_true.extend(targets.cpu().numpy())
    
    # 逆标准化和计算指标
    y_pred = scaler.inverse_transform(np.array(y_pred))
    y_true = scaler.inverse_transform(np.array(y_true))
    mse = mean_squared_error(y_true, y_pred)
    
    return train_losses, mse, total_time, avg_memory

# 主程序
if __name__ == "__main__":
    # 加载数据
    (X_train, y_train), (X_test, y_test), scaler = load_data(
        r'E:\毕业设计\data\ETT-small\ETTh1.csv'
    )
    
    # 创建DataLoader
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 初始化模型
    models = {
        "LSTM": LSTM(input_size=1, hidden_size=512, num_layers=1, output_size=1).to(device),
        "MinLSTM": MinLSTM(input_size=1, hidden_size=50, output_size=1).to(device)
    }
    
    # 训练和评估模型
    results = {}
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        train_losses, test_mse, total_time, avg_memory = train_and_evaluate(
            model, train_loader, test_loader, loss_fn, optimizer, num_epochs
        )
        results[name] = {
            "train_losses": train_losses,
            "test_loss": test_mse,
            "total_time": total_time,
            "avg_memory": avg_memory
        }
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    for name, result in results.items():
        train_losses = result["train_losses"]
        test_loss = result["test_loss"]
        
        # 绘制训练损失
        plt.plot(train_losses, label=f'{name} Train Loss')
        
        # 绘制测试损失（在最后一步）
        plt.axhline(y=test_loss, color='r', linestyle='--', label=f'{name} Test Loss ({test_loss:.4f})')
    
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss vs. Steps')
    plt.legend()
    plt.grid(True)
    plt.show()

# 新增：绘制时间和内存的柱状图
plt.figure(figsize=(12, 5))

# 训练时间比较
plt.subplot(1, 2, 1)
times = [results[name]['total_time'] for name in models]
bars_time = plt.bar(models.keys(), times, color=['blue', 'orange'])
plt.title('Total Training Time Comparison')
plt.ylabel('Seconds')

# 在柱子上显示具体数值
for bar in bars_time:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{height:.2f}s', ha='center', va='bottom')

# 内存消耗比较
plt.subplot(1, 2, 2)
mems = [results[name]['avg_memory'] for name in models]
bars_mem = plt.bar(models.keys(), mems, color=['blue', 'orange'])
plt.title('Average Memory Usage Comparison')
plt.ylabel('Memory Usage (MB)')

# 在柱子上显示具体数值
for bar in bars_mem:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{height:.2f}MB', ha='center', va='bottom')

plt.tight_layout()
plt.show()

