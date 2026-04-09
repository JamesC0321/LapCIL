import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy
def normal(x):
    x_min = torch.min(x)
    x_max = torch.max(x)
    x = (x - x_min) / (x_max - x_min)
    return x


def g_channel_Laplacian(x):
    b, c, w, h = x.shape
    avepool = nn.AdaptiveAvgPool2d((1, 1))
    x_g = avepool(x)  # [b, c, 1, 1]

    z_g_out = torch.Tensor().cuda() if x.is_cuda else torch.Tensor()

    for i in range(b):
        x_g_i = x_g[i].view(c)  # [c]
        _, index = torch.topk(x_g_i, 1)
        mu = float(index[0])  # Laplacian 的位置参数 μ

        # 设定 Laplacian 的尺度参数 b
        # 可以根据 channel 数动态设置，例如 b = c / 4
        b_scale = c / 4.0  # 你可以调整这个值

        # 构建 x 坐标：0 到 c-1（注意：torch.linspace(0, c, c) 实际上是 [0, ..., c] 共 c+1 个点）
        x_coords = torch.arange(c, dtype=torch.float32, device=x.device)  # shape: [c]

        # Laplacian PDF: (1/(2b)) * exp(-|x - mu| / b)
        laplace_pdf = (1.0 / (2.0 * b_scale)) * torch.exp(-torch.abs(x_coords - mu) / b_scale)

        # reshape to [1, c, 1, 1]
        z_g = laplace_pdf.view(1, c, 1, 1)
        z_g_out = torch.cat((z_g_out, z_g), dim=0)

    z_g_out = normal(z_g_out)
    return x * z_g_out
def g_channel(x):
    b, c, w, h = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    avepool = nn.AdaptiveAvgPool2d((1, 1))
    x_g = avepool(x)
    z_g_out = torch.Tensor().cuda()
    for i in range(b):
        x_g_i = x_g[i].view([c])
        _, index = torch.topk(x_g_i, 1)
        mean = index[0] * 1.0  # 均值
        covariance = (c / 2) ** 2  # 协方差
        gaussian_distribution = torch.distributions.Normal(mean, covariance)
        x_g_index = torch.linspace(0, c, c).cuda()
        z_g = gaussian_distribution.log_prob(x_g_index.flatten())
        z_g = z_g.exp().reshape(x_g_index.shape)
        z_g = z_g.view([1, c, 1, 1])
        z_g_out = torch.cat((z_g_out, z_g), dim=0)
    z_g_out = normal(z_g_out)
    return x * z_g_out

def g_channel_linear(x):
    x = torch.transpose(x,1,2)
    b, c, n = x.shape[0], x.shape[1], x.shape[2]

    x_g = torch.mean(x,dim=2)
    z_g_out = torch.Tensor().cuda()
    for i in range(b):
        x_g_i = x_g[i].view([c])
        _, index = torch.topk(x_g_i, 1)
        mean = index[0] * 1.0  # 均值
        covariance = (c / 2) ** 2  # 协方差
        gaussian_distribution = torch.distributions.Normal(mean, covariance)
        x_g_index = torch.linspace(0, c, c).cuda()
        z_g = gaussian_distribution.log_prob(x_g_index.flatten())
        z_g = z_g.exp().reshape(x_g_index.shape)
        z_g = z_g.view([1, c, 1])
        z_g_out = torch.cat((z_g_out, z_g), dim=0)
    z_g_out = normal(z_g_out)
    return x * z_g_out
def g_channel_linear_Laplacian(x):
    x = torch.transpose(x,1,2)
    b, c, n = x.shape[0], x.shape[1], x.shape[2]

    x_g = torch.mean(x,dim=2)
    z_g_out = torch.Tensor().cuda()
    for i in range(b):
        x_g_i = x_g[i].view(c)  # [c]
        _, index = torch.topk(x_g_i, 1)
        mu = float(index[0])  # Laplacian 的位置参数 μ

        # 设定 Laplacian 的尺度参数 b
        # 可以根据 channel 数动态设置，例如 b = c / 4
        b_scale = c / 4.0  # 你可以调整这个值

        # 构建 x 坐标：0 到 c-1（注意：torch.linspace(0, c, c) 实际上是 [0, ..., c] 共 c+1 个点）
        x_coords = torch.arange(c, dtype=torch.float32, device=x.device)  # shape: [c]

        # Laplacian PDF: (1/(2b)) * exp(-|x - mu| / b)
        laplace_pdf = (1.0 / (2.0 * b_scale)) * torch.exp(-torch.abs(x_coords - mu) / b_scale)

        # reshape to [1, c, 1, 1]
        z_g = laplace_pdf.view(1, c, 1)
        z_g_out = torch.cat((z_g_out, z_g), dim=0)
    z_g_out = normal(z_g_out)
    return x * z_g_out

def g_channel_linear_Triangular(x):
    """
    使用三角核（Triangular Kernel）的通道加权模块。
    Input: x of shape [B, N, C]  (e.g., N tokens, C channels)
    Output: weighted x of same shape
    """
    x = torch.transpose(x, 1, 2)  # [B, C, N]
    B, C, N = x.shape

    # 全局平均（沿 N 维度）
    x_g = torch.mean(x, dim=2)  # [B, C]

    # 预分配输出张量（避免 for 循环拼接）
    z_g_out = torch.empty(B, C, 1, 1, device=x.device)

    # 动态带宽：例如 b = C / k，k 可调（越大，影响越局部）
    bandwidth = C / 8  # 你可以调整这个值，如 4.0, 8.0, 16.0

    # 通道坐标：0 到 C-1
    x_coords = torch.arange(C, dtype=torch.float32, device=x.device)  # [C]

    for i in range(B):
        # 找到响应最强的通道索引
        _, top_idx = torch.topk(x_g[i], 1)  # [1]
        mu = float(top_idx.item())          # scalar

        # 三角核：max(0, 1 - |x - mu| / bandwidth)
        distances = torch.abs(x_coords - mu)           # [C]
        weights = F.relu(1.0 - distances / bandwidth)  # [C]

        z_g_out[i] = weights.view(1, C, 1)

    # 归一化（按你原来的 normal 函数）
    z_g_out = normal(z_g_out)  # [B, C, 1, 1]

    # 广播相乘
    return x * z_g_out  # [B, C, N]
def g_spatial(x):
    b, c, w, h = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    x_g = torch.mean(x, dim=1, keepdim=True)
    z_g_out = torch.Tensor().cuda()
    for i in range(b):
        x_g_i = x_g[i].view((w * h))
        _, index = torch.topk(x_g_i, 1)
        r_index = int(index[0] / w)
        c_index = int(index[0] - w * r_index)
        mean = torch.tensor([1.0 * c_index, 1.0 * r_index])  # 均值
        covariance = torch.tensor([[1.0 * (w / 2) ** 2, 0.0], [0.0, 1.0 * (h / 2) ** 2]])  # 协方差矩阵
        mean = mean.cuda()
        covariance = covariance.cuda()
        gaussian_distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance)
        x_g_index = torch.linspace(0, w, w).cuda()
        y_g_index = torch.linspace(0, h, h).cuda()
        x_g_index, y_g_index = torch.meshgrid(x_g_index, y_g_index)
        x_g_index = x_g_index.cuda()
        y_g_index = y_g_index.cuda()
        x_y_g_index = torch.stack([x_g_index.flatten(), y_g_index.flatten()], dim=1)
        x_y_g_index = x_y_g_index.cuda()
        z_g = gaussian_distribution.log_prob(x_y_g_index)
        z_g = z_g.exp().reshape(x_g_index.shape)
        z_g = z_g.view([1, 1, w, h])
        z_g_out = torch.cat((z_g_out, z_g), dim=0)
    z_g_out = normal(z_g_out)
    return x * z_g_out

def g_spatial_attention(x):
    b, c, w, h = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    x_g = torch.mean(x, dim=1, keepdim=True)
    z_g_out = torch.Tensor().cuda()
    for i in range(b):
        x_g_i = x_g[i].view((w * h))
        _, index = torch.topk(x_g_i, 1)
        r_index = int(index[0] / w)
        c_index = int(index[0] - w * r_index)
        mean = torch.tensor([1.0 * c_index, 1.0 * r_index])  # 均值
        covariance = torch.tensor([[1.0 * (w / 2) ** 2, 0.0], [0.0, 1.0 * (h / 2) ** 2]])  # 协方差矩阵
        mean = mean.cuda()
        covariance = covariance.cuda()
        gaussian_distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance)
        x_g_index = torch.linspace(0, w, w).cuda()
        y_g_index = torch.linspace(0, h, h).cuda()
        x_g_index, y_g_index = torch.meshgrid(x_g_index, y_g_index)
        x_g_index = x_g_index.cuda()
        y_g_index = y_g_index.cuda()
        x_y_g_index = torch.stack([x_g_index.flatten(), y_g_index.flatten()], dim=1)
        x_y_g_index = x_y_g_index.cuda()
        z_g = gaussian_distribution.log_prob(x_y_g_index)
        z_g = z_g.exp().reshape(x_g_index.shape)
        z_g = z_g.view([1, 1, w, h])
        z_g_out = torch.cat((z_g_out, z_g), dim=0)
    z_g_out = normal(z_g_out)
    return x * z_g_out, z_g_out

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def save_attention_heatmap_score(
    attn_scores,
    output_path='attention_heatmap.png',
    cmap='jet',
    alpha=0.6,
    figsize=(10, 10),
    dpi=300,
    normalize=True,
    return_topk=10,
    interpolation='nearest',
    save_raw_scores=False,
    save_as_csv=False,
    save_as_excel=False
):
    """
    使用 plt.imshow 绘制注意力热力图，并可选保存原始分数为 .npy / .csv / .xlsx。

    新增参数:
        save_raw_scores (bool): 保存 .npy（默认 True）
        save_as_csv (bool): 保存为 CSV 文件
        save_as_excel (bool): 保存为 Excel (.xlsx) 文件
    """
    # 转换为 NumPy
    if isinstance(attn_scores, torch.Tensor):
        attn = attn_scores.detach().cpu().numpy()
    else:
        attn = np.array(attn_scores)

    attn = attn.squeeze()
    if attn.ndim != 2:
        raise ValueError(f"注意力分数必须是2D，当前形状: {attn.shape}")

    original_attn = attn.copy()

    # 归一化用于可视化
    if normalize:
        vmin, vmax = attn.min(), attn.max()
        if vmax - vmin > 0:
            attn_vis = (attn - vmin) / (vmax - vmin)
        else:
            attn_vis = np.zeros_like(attn)
    else:
        attn_vis = attn

    # 绘图
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('white')
    ax.imshow(attn_vis, cmap=cmap, alpha=alpha, interpolation=interpolation)
    ax.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    ax.margins(0, 0)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig)
    print(f"热力图已保存至: {out_path}")

    # === 保存原始分数 ===
    stem = out_path.with_suffix('')  # 去掉 .png

    if save_raw_scores:
        np.save(stem.with_suffix('.npy'), original_attn)
        print(f"原始分数已保存为: {stem}.npy")

    if save_as_csv:
        df = pd.DataFrame(original_attn)
        csv_path = stem.with_suffix('.csv')
        df.to_csv(csv_path, index=False, header=False)
        print(f"原始分数已保存为 CSV: {csv_path}")

    if save_as_excel:
        df = pd.DataFrame(original_attn)
        excel_path = stem.with_suffix('.xlsx')
        df.to_excel(excel_path, index=False, header=False)
        print(f"原始分数已保存为 Excel: {excel_path}")

    # 返回 top-k
    if return_topk and return_topk > 0:
        flat = original_attn.ravel()
        topk_idx = np.argpartition(flat, -return_topk)[-return_topk:]
        topk_idx = topk_idx[np.argsort(-flat[topk_idx])]
        scores = flat[topk_idx].tolist()
        positions = [tuple(np.unravel_index(i, original_attn.shape)) for i in topk_idx]
        return {"scores": scores, "positions": positions}

    return None

def save_attention_heatmap(
        attn_scores,
        output_path='attention_heatmap.png',
        cmap='jet',
        alpha=0.6,
        figsize=(10, 10),
        dpi=300,
        normalize=True,
        return_topk=10
):
    """
    将注意力分数保存为热力图图像（白色背景，无坐标轴），并返回注意力最高的 top-k 位置。

    参数:
        attn_scores (torch.Tensor or np.ndarray):
            注意力分数，形状应为 [1, H, W] 或 [H, W]。
        output_path (str or Path):
            输出图像的保存路径（含文件名和扩展名）。
        cmap (str):
            热力图颜色映射，默认 'jet'。
        alpha (float):
            热力图透明度，范围 [0, 1]，默认 0.6。
        figsize (tuple):
            图像尺寸（英寸），默认 (10, 10)。
        dpi (int):
            图像分辨率，默认 300。
        normalize (bool):
            是否将注意力分数归一化到 [0, 1]，默认 True。
        return_topk (int or None):
            返回注意力分数最高的前 k 个位置；若为 None 则不返回。默认 10。

    返回:
        dict 或 None: 若 return_topk > 0，返回形如：
            {
                "scores": [0.98, 0.97, ..., 0.90],
                "positions": [(row0, col0), (row1, col1), ..., (row9, col9)]
            }
    """
    # 转换为 NumPy 数组
    if isinstance(attn_scores, torch.Tensor):
        attn = attn_scores.detach().cpu().numpy()
    else:
        attn = np.array(attn_scores)

    # 去除多余的维度（如 [1, H, W] -> [H, W]）
    attn = attn.squeeze()
    if attn.ndim != 2:
        raise ValueError(f"注意力分数必须是2D数组，当前形状为 {attn.shape}")

    original_attn = attn.copy()  # 保留原始值用于 top-k（无论是否归一化）

    # 归一化（仅用于可视化）
    if normalize:
        vmin, vmax = attn.min(), attn.max()
        if vmax - vmin != 0:
            attn = (attn - vmin) / (vmax - vmin)
        else:
            attn = np.zeros_like(attn)

    # 创建绘图
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('white')

    sns.heatmap(
        attn,
        cmap=cmap,
        cbar=False,
        alpha=alpha,
        zorder=2,
        linewidths=0,
        xticklabels=False,
        yticklabels=False,
        ax=ax
    )

    ax.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    ax.margins(0, 0)

    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 保存图像
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig)  # 释放内存

    print(f"注意力热力图已保存至: {output_path}")

    # 返回 top-k 分数和位置
    if return_topk and return_topk > 0:
        # 展平后取 top-k 索引
        flat_indices = np.argpartition(original_attn.ravel(), -return_topk)[-return_topk:]
        # 按分数从高到低排序
        topk_flat_indices = flat_indices[np.argsort(-original_attn.ravel()[flat_indices])]
        topk_scores = original_attn.ravel()[topk_flat_indices].tolist()
        topk_positions = [tuple(np.unravel_index(idx, original_attn.shape)) for idx in topk_flat_indices]

        return {
            "scores": topk_scores,
            "positions": topk_positions  # 格式: [(row, col), ...]
        }

    return None
if __name__ == "__main__":
    # data = torch.randn((1,52222,1024)).to("cuda")
    # out = g_channel_linear_Laplacian(data)
    # out = g_channel_linear(data)
    data = torch.randn((1,1024,64,64)).to("cuda")
    out,attention = g_spatial_attention(data)
    attention = attention.squeeze(0)
    result = save_attention_heatmap(attention,r'/media/chen/新加卷/EC_code/VMCIL/Model_ALTer_AAAI/Hatmap/MAP/att_map.png',return_topk=10)
    print("Top-10 注意力分数:", result["scores"])
    print("对应位置 (行, 列):", result["positions"])