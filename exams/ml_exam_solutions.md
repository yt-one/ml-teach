# 机器学习考试试卷 —— 参考答案与讲解

> 范围：机器学习入门、线性回归、SVM、决策树、随机森林等

---

## 一、选择题（每题 2 分，共 40 分）

1. 以下哪个不是机器学习的典型应用场景？  
   **答案：C**  
   讲解：浏览器渲染网页属于前端渲染/图形系统范畴，非典型 ML 任务。

2. 在监督学习中，训练数据包含：  
   **答案：B**  
   讲解：监督学习需要“输入特征 + 对应目标标签”。

3. 以下哪个算法是非参数化模型？  
   **答案：C（决策树）**  
   讲解：决策树不依赖固定参数维度，模型复杂度随数据增长而变。

4. 决策树在选择划分特征时，常用的指标是：  
   **答案：B（信息增益/基尼指数）**  
   讲解：分类常用信息增益/基尼；回归树常用 MSE / MAE。

5. 随机森林中每棵树的训练样本是如何获得的？  
   **答案：C（自举采样 Bootstrap）**  
   讲解：对训练集有放回采样得到每棵树的子样本集。

6. 训练集和测试集的主要区别是：  
   **答案：B**  
   讲解：训练集用于训练；测试集用于**最终**评估泛化性能。

7. 假阴性 (FN) 的含义是：  
   **答案：C（实际为正例，预测为负例）**

8. 以下哪个性能指标更适合类别极度不平衡的数据集？  
   **答案：B（Precision/Recall 更合适）**  
   讲解：不平衡时 Accuracy 不可靠；ROC AUC 有时也会被“极不平衡”掩盖，通常优先关注 **Precision/Recall**（或 PR AUC）。

9. 在线性可分情况下，SVM 的决策边界是：  
   **答案：B（最大化间隔的超平面）**

10. 在 SVM 中，参数 C 的作用是：  
    **答案：B（控制正则化强度/对误分类的惩罚）**  
    讲解：C 越大，越重视训练误差（更少容忍误分），等价于更弱正则。

11. 使用核 SVM 时，核函数的作用是：  
    **答案：B（映射到高维以实现线性可分）**

12. 批量梯度下降 (Batch GD) 中：  
    **答案：B（每次迭代用全部训练集更新）**

13. 随机梯度下降 (SGD) 的主要优点是：  
    **答案：B（迭代快、可处理大数据）**

14. 学习率调度的作用是：  
    **答案：B（逐渐降低学习率以保证收敛）**

15. Bagging 与 Boosting 的区别在于：  
    **答案：B（Bagging 并行，Boosting 串行）**

16. 在 AdaBoost 中，每一轮弱分类器的权重基于：  
    **答案：B（分类器的准确率/错误率）**

17. 梯度提升 (GBDT) 与 AdaBoost 的主要不同点：  
    **答案：A（GBDT 基于对损失的梯度下降，AdaBoost 基于分类错误率加权）**

18. 在梯度提升树中，学习率过小会导致：  
    **答案：B（欠拟合，训练时间变长）**

19. 随机森林相比单棵决策树的优势是：  
    **答案：C（泛化更好）**

20. 随机森林分裂时随机选特征子集的目的：  
    **答案：B（增加树之间差异，降低相关性）**

---

## 二、填空题（每题 2 分，共 20 分）

1. 机器学习的两个主要分支是 **监督学习** 和 **无监督学习**。  
2. 线性回归的目标是最小化 **均方误差（MSE）**。  
3. 岭回归通过在损失中加入 **L2** 范数作为正则项。  
4. Bagging 的全称是 **Bootstrap Aggregating**。  
5. 随机森林每次分裂时随机选择一个 **特征子集（feature subspace）**。  
6. Lasso 回归对不重要特征权重起到 **稀疏化/置零（特征选择）** 效果。  
7. 将二分类器扩展到多分类常用 **一对多（OvR）** 和 **一对一（OvO）**。  
8. SVM 回归的目标是让预测尽量落在 **\(\epsilon\)-不敏感带** 内。  
9. Recall（召回率）衡量的是 **实际为正的样本中被预测为正的比例**。  
10. 完整 ML 项目流程最后一步是 **部署与监控（上线与维护）**。

---

## 三、判断题（每题 2 分，共 10 分）

- KNN 属于参数化模型。（**错**）→ KNN 为**非参数化**。  
- 在极度不平衡数据集上，高准确率一定代表模型好。（**错**）  
- ROC 曲线下面积 (AUC) 越大，模型性能越好。（**对**）  
- 决策树越深越好，因为它可以拟合更多样本。（**错**，易过拟合）  
- 梯度提升中的每棵树都尝试拟合前一棵树的残差。（**对**）

---

## 四、简答题（每题 5 分，共 30 分）

### 1) 学习曲线（`learning_curve`）的含义与判别过拟合/欠拟合
- 含义：随着**训练样本数量**增加，给出训练分数与验证分数的曲线。  
- 判别：  
  - **高偏差/欠拟合**：训练分数低，验证分数低，二者差距小；再加数据几乎不提升。  
  - **高方差/过拟合**：训练分数高，验证分数低，二者差距大；可用正则/降复杂度/更多数据缓解。  
  - **数据不足**：训练/验证差距逐渐缩小但验证分数仍上升 → 继续加数据有益。

### 2) 特征量级差异对算法的影响与处理
- 受影响显著：基于距离/内积的算法（KNN、SVM（尤其 RBF/多项式核）、线性/逻辑回归（用 GD 训练）等）。  
- 影响：收敛慢、解被某些大尺度特征主导、决策边界畸变。  
- 处理：**标准化/归一化**（StandardScaler/MinMaxScaler）、对数/幂变换，必要时做**流水线**统一处理。

### 3) `LinearRegression.fit` 的实现思路 
- 利用 **SVD/最小二乘**（`np.linalg.lstsq` 或 scipy SVD）求解 \(\hat{\beta}=(X^TX)^{-1}X^Ty\) 的**数值稳定**实现；支持 `sample_weight`。  
- 无正则项；大规模数据常用 `SGDRegressor` 做近似。

### 4) 决策树为何易过拟合 & 缓解方法
- 原因：高方差；贪心划分可拟合噪声；深树可记忆训练集。  
- 缓解：**预剪枝**（限制 `max_depth/min_samples_split/min_samples_leaf/max_leaf_nodes/max_features`）、**集成学习**（随机森林、梯度提升）、更多数据/降噪。

### 5) 随机森林的特征重要性来源
- **基于不纯度减少（MDI）**：统计每棵树每次划分带来的 Gini/熵降低，按特征累加并对树平均。  
- **Permutation 重要性**：打乱单个特征，观察验证性能下降幅度。随机森林 因为多样化和 OOB 估计，重要性评估更稳健。

### 6) 弱学习器 vs 强学习器；为何集成能“变强”
- 弱学习器：性能**略优于随机**（如准确率 > 50% 的二分类）。  
- 强学习器：具有较高泛化性能。  
- 机制：Bagging **降方差**，Boosting **逐步纠错，提升性能/减少偏差**；独立/低相关的多个弱学习器经合理聚合可显著提升表现（投票/加权/堆叠学习器）。

---

## 五、综合题（每题 10 分，共 50 分）

### 1) 线性回归：California Housing 的 MSE
```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X, y = fetch_california_housing(return_X_y=True, as_frame=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 线性回归对尺度敏感，建议标准化（尤其含不同尺度特征时）
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std  = scaler.transform(X_test)

lin = LinearRegression()
lin.fit(X_train_std, y_train)
y_pred = lin.predict(X_test_std)
mse = mean_squared_error(y_test, y_pred)
print("LinearRegression MSE:", mse)
```

### 2) Softmax 回归在 Iris 上的 5 折准确率
```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X, y = load_iris(return_X_y=True)
clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
print("5-fold accuracies:", scores)
print("Mean accuracy:", scores.mean())
```

### 3) 结合数学公式：SVM 最大化间隔 ≈ L2 正则化
对硬间隔 SVM：最大化几何间隔 $\gamma = \frac{2}{\|w\|}$ 等价于最小化 $\frac{1}{2}\|w\|^2$。  
软间隔（允许误差）加入松弛变量 $\xi_i$：
$$
\min_{w,b,\xi} \; \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n \xi_i
\quad \text{s.t.}\; y_i(w^\top x_i + b) \ge 1 - \xi_i,\; \xi_i \ge 0.
$$
其中 $\frac{1}{2}\|w\|^2$ 即 **L2 正则项**，约束等价于 **hinge 损失** 的上界：
$$
\min_{w,b}\; \frac{1}{2}\|w\|^2 + C \sum_{i=1}^n \max\{0,\,1 - y_i(w^\top x_i + b)\}.
$$
因此，“最大化间隔”在优化上体现为对 $\|w\|$ 的惩罚（L2 正则），以获得更大的间隔和更好的泛化。

### 4) BaggingClassifier vs RandomForestClassifier（Iris）
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

X, y = load_iris(return_X_y=True)

base = DecisionTreeClassifier(random_state=42)
bag = BaggingClassifier(base_estimator=base, n_estimators=100, random_state=42, n_jobs=-1)
rf  = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

bag_scores = cross_val_score(bag, X, y, cv=5, scoring="accuracy")
rf_scores  = cross_val_score(rf,  X, y, cv=5, scoring="accuracy")

print("Bagging 5-fold acc:", bag_scores.mean())
print("RandomForest 5-fold acc:", rf_scores.mean())
```
讲解：两者都用 Bagging 思想，**RF 还在每次分裂引入特征子采样**，进一步减少树间相关性，通常略优于“Bag + 决策树”。

### 5) 递归手写决策树（基尼指数）
```python
import numpy as np

tree = []  # 仅用于记录打印/检查，不是必须的数据结构

def gini_impurity(y):
    """Gini(y) = 1 - sum_k p_k^2"""
    m = len(y)
    if m == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / m
    return 1.0 - np.sum(p ** 2)

def best_split(X, y):
    """在所有特征与候选阈值上，寻找使基尼指数最小的二分划分。
       返回: (best_feature, best_threshold, best_gini, left_indices, right_indices)
    """
    m, n = X.shape
    if m <= 1:
        return None, None, None, None, None

    parent_gini = gini_impurity(y)
    best_gain = 0.0
    best_feat, best_thr = None, None
    best_left_idx, best_right_idx = None, None

    for feat in range(n):
        # 按该特征排序，尝试相邻值的中点作为阈值
        sorted_idx = np.argsort(X[:, feat])
        Xf, yf = X[sorted_idx, feat], y[sorted_idx]

        # 候选阈值：相邻不同值的中点
        for i in range(1, m):
            if Xf[i] == Xf[i-1]:
                continue
            thr = 0.5 * (Xf[i] + Xf[i-1])
            left_mask  = X[:, feat] <= thr
            right_mask = ~left_mask
            y_left, y_right = y[left_mask], y[right_mask]
            g_left  = gini_impurity(y_left)
            g_right = gini_impurity(y_right)
            g = (len(y_left) * g_left + len(y_right) * g_right) / m
            gain = parent_gini - g
            if gain > best_gain:
                best_gain = gain
                best_feat, best_thr = feat, thr
                best_left_idx = np.where(left_mask)[0]
                best_right_idx = np.where(right_mask)[0]

    return best_feat, best_thr, parent_gini - best_gain, best_left_idx, best_right_idx

def majority_class(y):
    vals, counts = np.unique(y, return_counts=True)
    return vals[np.argmax(counts)]

def build_tree_recursive(X, y, node_indices, max_depth, current_depth):
    """递归构建树（按基尼指数选择最佳划分）"""
    X_node, y_node = X[node_indices], y[node_indices]
    node_id = len(tree)
    # 记录当前节点
    record = {
        "node_id": node_id,
        "depth": current_depth,
        "n_samples": len(node_indices),
        "gini": gini_impurity(y_node),
    }

    # 停止条件：达到最大深度或纯节点
    if current_depth == max_depth or gini_impurity(y_node) == 0.0:
        record["type"] = "leaf"
        record["class"] = int(majority_class(y_node))
        tree.append(record)
        return

    # 寻找最佳划分
    feat, thr, _, left_idx, right_idx = best_split(X_node, y_node)
    if feat is None or left_idx.size == 0 or right_idx.size == 0:
        record["type"] = "leaf"
        record["class"] = int(majority_class(y_node))
        tree.append(record)
        return

    record["type"] = "split"
    record["feature"] = int(feat)
    record["threshold"] = float(thr)
    tree.append(record)

    # 递归构建左右子树（注意将局部索引映射回全局索引）
    left_global  = node_indices[left_idx]
    right_global = node_indices[right_idx]
    build_tree_recursive(X, y, left_global,  max_depth, current_depth + 1)
    build_tree_recursive(X, y, right_global, max_depth, current_depth + 1)

# 使用示例：
# X = np.array([[2.7, 2.5], [1.3, 3.0], [3.1, 1.9], [2.0, 2.7], [1.0, 1.0]])
# y = np.array([0, 1, 0, 0, 1])
# node_indices = np.arange(len(y))
# build_tree_recursive(X, y, node_indices, max_depth=3, current_depth=0)
# print(tree)
```
要点：
- 计算 **Gini**，遍历所有特征与候选阈值（相邻值中点），取信息增益最大的划分。  
- 停止条件：到达 `max_depth` 或纯节点。  
- 未做性能优化（如直方图划分、特征采样等）。