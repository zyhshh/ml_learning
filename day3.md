决策树算法梳理
1. 信息论基础(熵,联合熵,条件熵 信息增益 基尼不纯度)
2. 决策树的不同分类算法(ID3算法,C4.5,CART分类树)的原理及应用场景
3. 回归树原理
4. 决策树防止过拟合手段
5. 评估模型
6. sklearn参数详解,python绘制决策树
1. 信息论基础(熵,联合熵,条件熵 信息增益 基尼不纯度)
1.熵：本是热力学中表征物质状态的参量之一，用符号S表示，其物理意义是体系混乱程度的度量。对于机器学习算法来说，熵指代香农熵，是一种不确定性度量。它是表示随机变量不确定的度量，是对所有可能发生的事件产生的信息量的期望。对于事件X，有n种可能结果，且概率分别为p1，p2，...，pn，则熵H(X)为：



基本性质：

均匀分布具有最大的熵。一个好的不确定性度量会在均匀分布时达到最大的值。给定n个可能的结果，在所有结果的概率相同时得到最大的熵。
对于独立事件，熵是可加的。两个独立事件的联合熵等于各个独立事件的熵的和。
具有非零概率的结果数量增加，熵也会增加。加入发生概率为0的结果并不会有影响。
连续性。不确定性度量应该是连续的，熵函数是连续的。
具有更多可能结果的均匀分布有更大的不确定性。
非负性。事件拥有非负的不确定性。
确定事件的熵为0。有确定结果的事件具有0不确定性。
参数排列不变性。调转参数顺序没有影响。
2.联合熵：一维随机变量分布推广到多维随机变量分布，则其联合熵 (Joint entropy) 为：



注：熵只依赖于随机变量的分布，与随机变量取值无关。

3.条件熵： H(Y|X) 表示在已知随机变量 X 的条件下随机变量 Y 的不确定性。条件熵 H(Y|X) 定义为 X 给定条件下 Y 的条件概率分布的熵对  X 的数学期望。



条件熵 H(Y|X) 相当于联合熵 H(X,Y) 减去单独的熵 H(X)，即H(Y|X)=H(X,Y)−H(X)。证明：



当已知 H(X) 这个信息量的时候，H(X,Y) 剩下的信息量就是条件熵，描述 X 和 Y 所需的信息是描述 X 自己所需的信息,加上给定  X 的条件下具体化  Y 所需的额外信息。

4.信息增益：以某特征划分数据集前后的熵的差值。即待分类集合的熵和选定某个特征的条件熵之差（这里只的是经验熵或经验条件熵，由于真正的熵并不知道，是根据样本计算出来的），公式如下：



注：这里不要理解偏差，因为上边说了熵是类别的，但是在这里又说是集合的熵，没区别，因为在计算熵的时候是根据各个类别对应的值求期望来等到熵

5.基尼不纯度：将来自集合中的某种结果随机应用于集合中某一数据项的预期误差率。即从一个数据集中随机选取子项，度量其被错误的划分到其他组里的概率。



（1）显然基尼不纯度越小，纯度越高，集合的有序程度越高，分类的效果越好；

（2）基尼不纯度为 0 时，表示集合类别一致；

（3）基尼不纯度最高（纯度最低）时，



例，如果集合中的每个数据项都属于同一分类，此时误差率为 0。如果有四种可能的结果均匀地分布在集合中，此时的误差率为 1−0.25=0.75；

2. 决策树的不同分类算法(ID3算法,C4.5,CART分类树)的原理及应用场景 
ID3算法
 
 
  ID3算法的核心是在决策树各个结点上应用信息增益准则选择特征，递归地构建决策树。
  具体方法是：从根节点（root node）开始，对结点计算所有可能的特征信息增益，选择信息增益最大的特征作为结点特征，由该特征的不同取值建立子结点；再对子结点递归地调用以上方法，构建决策树；直到所有特征的信息均很小或没有特征可以选择为止。
 
 
C4.5算法
 
  C4.5算法与ID3算法相似，C4.5算法对ID3算法进行了改进。C4.5在生成过程中，用信息增益比来选择特征。
 
CART分类树
 
  CART是在给定输入随机变量X  
条件下输出随机变量Y的条件概率分布的学习方法。  
  CART算法由一下两步组成：  
    1）决策树生成：基于训练数据集生成决策树，生成的决策树要尽量大；  
    2）决策树剪枝：用验证数据集对已生成的树进行剪枝病选择最优子树，这时用损失函数最小作为剪枝的标准。  
 
应用场景  
 
  银行贷款申请、房产开发商房子的选址。  
3. 回归树原理
  决策树的生成就是递归地构建二叉决策树的过程，对于回归树用平方误差最小化准则。  
4.决策树防止过拟合手段  
  剪枝是决策树学习算法对付过拟合的主要手段。决策树剪枝的基本策略有“预剪枝”和“后剪枝”。  
  预剪枝是指在决策树生成过程中，对每个结点在划分前先进行估计，若当前结点的划分不能带来决策树泛化性能提升，则停止划分并将当前结点标记为叶节点。
  后剪枝是指先从训练集生成一颗完整的决策树，然后自底向上地对非结点进行考察，若讲该结点对应的子树替换成叶节点能带来决策树泛化性能的提升，则将该子树替换成叶节点。  
5. 模型评估
“所有模型都是坏的，但有些模型是有用的”。建立模型之后，接下来就要去评估模型，以确定此模型是否“有用”。sklearn库的metrics模块提供各种评估方法，包括分类评估、回归评估、聚类评估和交叉验证等，本节主要介绍分类模型评估方法。  
  评估分类是判断预测值时否很好的与实际标记值相匹配。正确的鉴别出正样本（True Positives）或者负样本（True Negatives）都是True。同理，错误的判断正样本（False Positive，即一类错误）或者负样本（False Negative，即二类错误）。  
  注意：True和False是对于评价预测结果而言，也就是评价预测结果是正确的(True)还是错误的(False)。而Positive和Negative则是样本分类的标记。



6. sklearn参数详解,python绘制决策树  
class sklearn.tree.DecisionTreeClassifier(criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
  参数详解：
  criterion=’gini’, string, optional (default=”gini”)，衡量分支好坏的标准  
  splitter=’best’, string, optional (default=”best”)，选择分支的策略  
  max_depth=None, int or None, optional (default=None)，树的最大深度  
  min_samples_split=2, int, float, optional (default=2)，分支时最小样本数  
  min_samples_leaf=1, int, float, optional (default=1)，叶子最少样本  
  min_weight_fraction_leaf=0.0, float, optional (default=0.)，叶子结点的最小权重  
  max_features=None, int, float, string or None, optional (default=None)，生成树时考虑的最多特征点数  
  random_state=None,  int, RandomState instance or None, optional (default=None)，打乱样本时所用的随机种子  
  max_leaf_nodes=None,  int or None, optional (default=None)，生成树时采用的最大叶子结点  
  min_impurity_decrease=0.0, float, optional (default=0.)，当产生分支时，增加的纯度  
  min_impurity_split=None,  float, (default=1e-7)，树停止生长的阈值  
  class_weight=None,  dict, list of dicts, “balanced” or None, default=None，分支权重预设定  
  presort=False，bool, optional (default=False)，提前对数据排序，加快树的生成  


