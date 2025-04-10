import torch
# 
# def generate_synthetic_task_vector(task_vector: torch.Tensor = None, num_synthetic: int = 10):
#     """
#     MDCS：Multi-Dimensional Consistency Scoring
#     """
#     # 确保输入是二维张量 (n_clients, n_dimensions)
#     assert task_vector.dim() == 2, "task_vector must be a 2D tensor"
# 
#     # 计算每个维度的均值和标准差
#     mean = torch.mean(task_vector, dim=0)  # (n_dimensions,)
#     std = torch.std(task_vector, dim=0)    # (n_dimensions,)
# 
#     # 避免除以零
#     std[std == 0] = 1e-8
# 
#     # 计算每个客户端的 Z-score
#     z_scores = (task_vector - mean) / std  # (n_clients, n_dimensions)
# 
#     # 计算每个客户端的 MDCS 评分（Z-score 绝对值之和）
#     mdc_scores = torch.sum(torch.abs(z_scores), dim=1)  # (n_clients,)
# 
#     # 找到 MDCS 评分最大的客户端索引
#     max_mdc_index = torch.argmax(mdc_scores)
# 
#     # 计算去除最大 MDCS 评分的均值
#     mean_update = torch.mean(torch.cat((task_vector[:max_mdc_index], task_vector[max_mdc_index+1:]), dim=0), dim=0, keepdim=True)  # (1, n_dimensions)
# 
#     # 生成合成更新
#     synthetic_updates = mean_update.repeat(num_synthetic, 1)
# 
#     # 将合成更新添加到原始任务向量中
#     augmented_task_vector = torch.cat((task_vector, synthetic_updates), dim=0)
# 
#     return augmented_task_vector, max_mdc_index



def cyclic_shift(vector: torch.Tensor, shift: int):
    """
    循环移动向量。
    """
    return torch.cat((vector[-shift:], vector[:-shift]))

def generate_synthetic_task_vector(task_vector: torch.Tensor = None, num_synthetic: int = 10):
    """
    MTCMS：Multi-Task Cyclic Model Shifting
    """
    # 确保输入是二维张量 (n_clients, n_dimensions)
    assert task_vector.dim() == 2, "task_vector must be a 2D tensor"

    # 计算每个维度的均值和标准差
    mean = torch.mean(task_vector, dim=0)  # (n_dimensions,)
    std = torch.std(task_vector, dim=0)    # (n_dimensions,)

    # 避免除以零
    std[std == 0] = 1e-8

    # 计算每个客户端的 Z-score
    z_scores = (task_vector - mean) / std  # (n_clients, n_dimensions)

    # 计算每个客户端的 MDCS 评分（Z-score 绝对值之和）
    mdc_scores = torch.sum(torch.abs(z_scores), dim=1)  # (n_clients,)

    # 选择 MDCS 评分最高的客户端索引
    max_mdc_index = torch.argmax(mdc_scores)

    # 选择0作为MDCS评分最高的客户端索引
    max_mdc_index = 0


    # 选择一个任务向量进行循环移动
    selected_vector = task_vector[max_mdc_index]
    vector_dim = selected_vector.shape[0]

    # 生成合成更新，使用循环移动
    synthetic_updates = []
    shift_value = torch.randint(1, vector_dim//1000, (1,)).item() # 随机选择一个循环移动的步数
    print(shift_value,"*******************")
    synthetic_vector = cyclic_shift(selected_vector, shift_value)
    for _ in range(num_synthetic):
        synthetic_updates.append(synthetic_vector.unsqueeze(0))
    
    synthetic_updates = torch.cat(synthetic_updates, dim=0)

    # 将合成更新添加到原始任务向量中
    augmented_task_vector = torch.cat((task_vector, synthetic_updates), dim=0)

    return augmented_task_vector, max_mdc_index
