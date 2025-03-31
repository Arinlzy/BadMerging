import torch

def generate_synthetic_task_vector(task_vector: torch.Tensor = None, num_synthetic: int = 10):
    """
    MDCS：Multi-Dimensional Consistency Scoring
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

    # 选择 MDCS 评分最小的客户端
    min_mdc_index = torch.argmin(mdc_scores)

    # 生成合成更新
    synthetic_updates = task_vector[min_mdc_index].unsqueeze(0).repeat(num_synthetic, 1)

    # 将合成更新添加到原始任务向量中
    augmented_task_vector = torch.cat((task_vector, synthetic_updates), dim=0)

    return augmented_task_vector, min_mdc_index