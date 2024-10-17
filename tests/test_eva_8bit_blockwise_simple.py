import torch
import bitsandbytes as bnb

def test_adam8bit_blockwise():
    # 创建一些随机数据和目标
    input_data = torch.randn(32, 1000).half().cuda()
    target = torch.randint(0, 10, (32,)).cuda()

    # 定义简单的全连接层
    model = torch.nn.Linear(1000, 10).half().cuda()

    # 定义损失函数和8-bit Adam优化器
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = bnb.optim.Eva8bit(model, lr=1e-3)

    # 前向传播
    output = model(input_data)
    loss = loss_fn(output, target)

    # 反向传播并更新参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss after optimization: {loss.item()}")

if __name__ == "__main__":
    test_adam8bit_blockwise()
