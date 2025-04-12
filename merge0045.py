import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 参数配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 24  # 用过去24小时预测未来
BATCH_SIZE = 128
num_epochs = 10

# 1. 数据加载与预处理（仅划分训练集和测试集）
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
    for i in range(len(data_scaled) - seq_len):
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

# 2. 创建数据加载器
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {'input': self.X[idx], 'target': self.y[idx]}

# 3. 标准GRU模型定义
class GRUModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播GRU
        out, _ = self.gru(x, h0)
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        
        # 通过全连接层
        out = self.fc(out)
        return out

# 4. MinGRU模型定义
class MinGRUForTimeSeries(nn.Module):
    def __init__(self, input_size=1, hidden_size=int, output_size=1, num_layers=1, device=device):
        super(MinGRUForTimeSeries, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.linear_z = nn.ModuleList()
        self.linear_h = nn.ModuleList()

        self.linear_z.append(nn.Linear(input_size, hidden_size))
        self.linear_h.append(nn.Linear(input_size, hidden_size))

        for _ in range(1, num_layers):
            self.linear_z.append(nn.Linear(hidden_size, hidden_size))
            self.linear_h.append(nn.Linear(hidden_size, hidden_size))

        self.final_proj = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def log_g(self, x):
        eps = 1e-8
        return torch.log(torch.tanh(x) + eps)

    def parallel_scan(self, log_coeffs, log_values):
        a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
        log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
        log_h = a_star + log_h0_plus_b_star
        return torch.exp(log_h)[:, 1:, :]

    def forward(self, x):
        batch_size, seq_len, input_size = x.shape
        h_0 = [torch.zeros((batch_size, 1, self.hidden_size), device=self.device)
               for _ in range(self.num_layers)]
        layer_input = x

        for layer in range(self.num_layers):
            k = self.sigmoid(self.linear_z[layer](layer_input))
            log_z = -F.softplus(-k)
            log_coeffs = -F.softplus(k)
            log_h_0 = self.log_g(h_0[layer])
            log_h_wavy = self.log_g(self.sigmoid(self.linear_h[layer](layer_input)))
            log_values = torch.cat([log_h_0, log_z + log_h_wavy], dim=1)
            h = self.parallel_scan(log_coeffs, log_values)
            layer_input = h

        final_hidden = h[:, -1, :]
        output = self.final_proj(final_hidden)
        return output
import time
# 5. 训练和评估函数（添加损失记录和时间记录）
def train_and_evaluate(model, train_dataloader, test_dataloader,
                      loss_fn, optimizer, num_epochs=10):
    train_losses = []
    test_losses = []
    memory_usage = []  # 新增内存记录
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_train_losses = []
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
            
            # 记录内存使用（仅CUDA）
            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # 转换为MB
                memory_usage.append(peak_mem)
            
            # 记录损失
            epoch_train_losses.append(loss.item())
            train_losses.append(loss.item())
            
            if idx % 100 == 0:
                print(f'Epoch: {epoch}, Step: {idx}, Train Loss: {loss.item():.4f}')

        # 测试阶段
        model.eval()
        epoch_test_losses = []
        with torch.no_grad():
            for batch in test_dataloader:
                inputs = batch['input']
                targets = batch['target']
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                epoch_test_losses.append(loss.item())
        
        # 计算平均内存（仅训练阶段）
        avg_memory = np.mean(memory_usage[-len(epoch_train_losses):]) if memory_usage else 0
        
        # 记录测试损失
        avg_test_loss = np.mean(epoch_test_losses)
        num_train_batches = len(epoch_train_losses)
        test_losses.extend([avg_test_loss] * num_train_batches)
        
        print(f'Epoch: {epoch} => Test Loss: {avg_test_loss:.4f} | Avg Memory: {avg_memory:.2f} MB')
    
    total_time = time.time() - start_time
    return train_losses, test_losses, total_time, np.mean(memory_usage) if memory_usage else 0

# 修改后的绘图函数
def plot_loss_curves(models_data, max_steps=1000):
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    line_styles = ['-', '--']
    
    for idx, (model_name, train_losses, test_losses, _, _) in enumerate(models_data):
        # 截取数据
        train_trunc = train_losses[:max_steps]
        test_trunc = test_losses[:max_steps]
        final_test_loss = test_losses[-1] if len(test_losses) > 0 else 0  # 获取最终测试损失
        
        # 绘制曲线
        plt.plot(train_trunc, label=f'{model_name} (Train)', 
                color=colors[idx], linestyle=line_styles[0], alpha=0.7)
        plt.plot(test_trunc, label=f'{model_name} (Test)', 
                color=colors[idx], linestyle=line_styles[1], alpha=0.7)
        plt.axhline(final_test_loss, color=colors[idx], linestyle=':', 
                   label=f'{model_name} Final Test: {final_test_loss:.4f}')
    
    plt.title('Training and Test Loss Comparison (First 1000 Steps)')
    plt.xlabel('Training Steps')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(r'E:\毕业设计\group\minGRU\step-loss.png', dpi=300)
    plt.show()



# 新增资源消耗可视化函数
def plot_resource_usage(results, save_path):
    names = [res[0] for res in results]
    times = [res[3] for res in results]
    memories = [res[4] for res in results]

    plt.figure(figsize=(12, 5))
    
    # 训练时间对比
    plt.subplot(1, 2, 1)
    bars = plt.bar(names, times, color=['skyblue', 'salmon'])
    plt.title('Total Training Time Comparison')
    plt.ylabel('Seconds')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom')
    
    # 内存消耗对比
    plt.subplot(1, 2, 2)
    bars = plt.bar(names, memories, color=['skyblue', 'salmon'])
    plt.title('Average Memory Usage Comparison')
    plt.ylabel('Memory Usage (MB)')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} MB',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# 主程序（保持数据加载和模型定义不变）
if __name__ == "__main__":
    # 数据加载
    (X_train, y_train), (X_test, y_test), scaler = load_data(r'E:\毕业设计\data\ETT-small\ETTh1.csv')
    
    # 创建数据加载器
    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=BATCH_SIZE)
    
    # 模型配置
    models = [
        {'name': 'Standard GRU', 'model': GRUModel(1, 500, 1).to(device), 'lr': 0.001},
        {'name': 'MinGRU', 'model': MinGRUForTimeSeries(hidden_size=500).to(device), 'lr': 0.001}
    ]
    
    # 训练和评估
    results = []
    for config in models:
        model = config['model']
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        print(f"\n=== Training {config['name']} ===")
        train_loss, test_loss, total_time, avg_memory = train_and_evaluate(
            model, train_loader, test_loader, nn.MSELoss(), optimizer, num_epochs)
        results.append( (config['name'], train_loss, test_loss, total_time, avg_memory) )
    
    # 绘制损失曲线
    plot_loss_curves(results)
    
    # 绘制资源消耗图
    plot_resource_usage(results, r'E:\毕业设计\group\minGRU\resource_usage.png')
    
    # 打印最终结果
    print("\n=== Final Results ===")
    for name, _, _, time, mem in results:
        print(f"{name}:")
        print(f"  Training Time: {time:.2f}s")
        print(f"  Avg Memory: {mem:.2f} MB\n")