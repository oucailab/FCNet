import numpy as np
import matplotlib.pyplot as plt


# 加载数据
a = np.load("AMAP_mask.npy")
b = np.load("land_mask.npy")

# 打印数据的总和
print("Sum of a:", np.sum(a))
print("Sum of b:", np.sum(b))

# 创建一个新的图形
plt.figure(figsize=(12, 6))

# 绘制 a 的灰度图
plt.subplot(1, 2, 1)
plt.imshow(a, cmap="gray")
plt.title("AMAP Mask")
plt.colorbar(orientation="horizontal")

# 绘制 b 的灰度图
plt.subplot(1, 2, 2)
plt.imshow(b, cmap="gray")
plt.title("Land Mask")
plt.colorbar(orientation="horizontal")

# 保存图像
plt.savefig("masks.png", bbox_inches="tight", dpi=300)
