# 强化学习C++实现

**tips:目前算法逻辑可读性较好，但使用时较为一般，有待优化**

**使用步骤：**

1. 配置c++环境，使用VS
2. 安装相关依赖库，尤其需要利用python环境配置matplotlib cpp用于绘图
3. src.h中设置相关参数，在main.cpp中调用函数计算结果
4. 部分结果涉及到修改函数中内部返回值（如截断式策略迭代计算误差，需要把boe.cpp中turuncated_policy_iteration返回的Vlist修改为errors，或增加一个返回值）
5. 其他可能存在的问题

## 详细步骤

### 1. C++环境配置

+ 若使用VSCode或其他平台，推荐配置gcc/g++编译器，具体步骤参考网络资源
+ 本项目使用Visual Studio(VS)编写，编译器为自带MSVC

### 2. 工具准备

原教程链接：https://zhuanlan.zhihu.com/p/585302210. 知乎@疯狂学习GIS

#### 2.1 包管理器vcpkg安装

vcpkg是微软提供给VS的包管理器，可以方便的对C++各种第三方库进行安装与管理。后续需要使用部分第三方包，因此需要进行vcpkg的安装。

首先，选定一个路径作为`vcpkg`的保存路径；随后，在这一文件夹下，按下`Shift`按钮并同时右击鼠标，选择“**在此处打开Powershell窗口**”。

![https://picx.zhimg.com/80/v2-584eca7e7a0c92a271b5b55548d75d27_1440w.webp?source=1def8aca](https://picx.zhimg.com/80/v2-584eca7e7a0c92a271b5b55548d75d27_1440w.webp?source=1def8aca)

随后，将弹出如下所示的窗口。

![https://pic1.zhimg.com/80/v2-c26acf1c3870ade7160d78a77d6887de_1440w.webp?source=1def8aca](https://pic1.zhimg.com/80/v2-c26acf1c3870ade7160d78a77d6887de_1440w.webp?source=1def8aca)

接下来，在其中输入如下的代码，并运行。（这里需要配置git）

> 注：也可以直接官网下载vcpkg

`git clone https://github.com/microsoft/vcpkg`

具体如下图所示。

![https://picx.zhimg.com/80/v2-dbe897627dd4175c03bc0dac9fadd78f_1440w.webp?source=1def8aca](https://picx.zhimg.com/80/v2-dbe897627dd4175c03bc0dac9fadd78f_1440w.webp?source=1def8aca)

稍等片刻，出现如下所示的界面，说明`vcpkg`安装完毕。

![https://picx.zhimg.com/80/v2-96a021905934b9ed334afdfa24d0978c_1440w.webp?source=1def8aca](https://picx.zhimg.com/80/v2-96a021905934b9ed334afdfa24d0978c_1440w.webp?source=1def8aca)

随后，输入如下代码，进入`vcpkg`保存路径。

`cd vcpkg`

再输入如下代码，激活`vcpkg`环境。

`.\bootstrap-vcpkg.bat`

具体如下图所示。

![https://picx.zhimg.com/80/v2-8699c3ddef9e3ebcbdb3aa3b8025571d_1440w.webp?source=1def8aca](https://picx.zhimg.com/80/v2-8699c3ddef9e3ebcbdb3aa3b8025571d_1440w.webp?source=1def8aca)

运行完毕后，将得到如下所示的结果。

![https://pic1.zhimg.com/80/v2-d2cdf56487a2fe4bd2bad8d188443639_1440w.webp?source=1def8aca](https://pic1.zhimg.com/80/v2-d2cdf56487a2fe4bd2bad8d188443639_1440w.webp?source=1def8aca)

接下来，再输入如下所示的代码，将`vcpkg`与我们的**Visual Studio**软件相连接。

`.\vcpkg integrate install`

具体如下图所示。

![https://pica.zhimg.com/80/v2-807c7939d86dbd9ec170aebf551a2984_1440w.webp?source=1def8aca](https://pica.zhimg.com/80/v2-807c7939d86dbd9ec170aebf551a2984_1440w.webp?source=1def8aca)

代码运行完毕后，如下图所示。

![https://pica.zhimg.com/80/v2-966309f86b719f25920599153d4b9426_1440w.webp?source=1def8aca](https://pica.zhimg.com/80/v2-966309f86b719f25920599153d4b9426_1440w.webp?source=1def8aca)

#### 2.2 matplotlib-cpp库配置

**在Visual Studio中配置matplotlib-cpp库，实现C++代码调用Python matplotlib接口进行绘图**

**1. matplotlib安装**

进入下载vcpkg的目录（包含vcpkg.exe文件），即可直接安装完成

```cpp
vcpkg.exe install matplotlib-cpp:x64-windows
```

或：

首先，依然在刚刚的界面中，输入如下代码，安装`matplotlibcpp`库。

`.\vcpkg install matplotlib-cpp`

代码运行结束后，得到如下所示的结果。

![https://pic1.zhimg.com/80/v2-35c6c729f1ccd20a1101d3e0535fbda7_1440w.webp?source=1def8aca](https://pic1.zhimg.com/80/v2-35c6c729f1ccd20a1101d3e0535fbda7_1440w.webp?source=1def8aca)

随后，输入如下所示的代码，安装64位的`matplotlibcpp`库。

`.\vcpkg install matplotlib-cpp:x64-windows`

运行代码后，得到如下所示的结果。

![https://picx.zhimg.com/80/v2-35b4ead5f9b0062534829e97e2891758_1440w.webp?source=1def8aca](https://picx.zhimg.com/80/v2-35b4ead5f9b0062534829e97e2891758_1440w.webp?source=1def8aca)

> 注: 使用vcpkg安装其他库也是上述过程，因此涉及到的其他库不再介绍安装过程

**2. matplotlibcpp配置**

首先，在刚刚配置的`vcpkg`的保存路径中，通过以下路径，找到`matplotlibcpp.h`文件，并将其打开。

![https://picx.zhimg.com/80/v2-03fcc775ee09d08575706c3a0c10b1d1_1440w.webp?source=1def8aca](https://picx.zhimg.com/80/v2-03fcc775ee09d08575706c3a0c10b1d1_1440w.webp?source=1def8aca)

随后，在其`#include`部分的最下方，添加如下代码。

`#include <string>` //这一步是为了支持string，根据需求添加

具体如下图所示。

![https://pica.zhimg.com/80/v2-e336181f18ccfe0e18869a9945943304_1440w.webp?source=1def8aca](https://pica.zhimg.com/80/v2-e336181f18ccfe0e18869a9945943304_1440w.webp?source=1def8aca)

同时，在该文件`340`行左右，将`template`开头的两行注释掉，如下图所示。

**必须注释，否则报错**

![https://picx.zhimg.com/80/v2-d05b2f42d13ecae5594ad0ad5d2c865b_1440w.webp?source=1def8aca](https://picx.zhimg.com/80/v2-d05b2f42d13ecae5594ad0ad5d2c865b_1440w.webp?source=1def8aca)

由于`matplotlibcpp`库是通过调用**Python**接口，实现在**C++** 代码中通过`matplotlib`库的命令绘制各类图像，因此配置`matplotlibcpp`库时还需要保证电脑中拥有**Python**环境。而这里的**Python**环境也有一个具体的要求——需要具有`Debug`版本的**Python**。

**注：Debug版本的Python需要在安装时选择，目前暂不清楚使用Anconda如何实现，因此直接安装Python**

**3. Python安装**

我们在**Python**的[官方下载地址](https://link.zhihu.com/?target=https%3A//www.python.org/downloads/)（[https://www.python.org/downloads/](https://link.zhihu.com/?target=https%3A//www.python.org/downloads/)）中下载

Python版本选择3.12以下的，由于后面的numpy库版本不能太高，对python支持版本也不高，因此最好选择低版本。

![https://picx.zhimg.com/80/v2-61727d975dc75620fc0a81132ad7bad4_1440w.webp?source=1def8aca](https://picx.zhimg.com/80/v2-61727d975dc75620fc0a81132ad7bad4_1440w.webp?source=1def8aca)

随后，双击打开刚刚下载好的安装包。

首先，选择“**Customize [installation](https://zhida.zhihu.com/search?content_id=533334331&content_type=Answer&match_order=1&q=installation&zhida_source=entity)**”选项。

![https://pica.zhimg.com/80/v2-571ba58a1f56931273ac2e9fb737eda5_1440w.webp?source=1def8aca](https://pica.zhimg.com/80/v2-571ba58a1f56931273ac2e9fb737eda5_1440w.webp?source=1def8aca)

接下来的页面，选择默认的配置即可。

![https://picx.zhimg.com/80/v2-8992fd54bc0a0c6fa234a04de591d0a8_1440w.webp?source=1def8aca](https://picx.zhimg.com/80/v2-8992fd54bc0a0c6fa234a04de591d0a8_1440w.webp?source=1def8aca)

随后的页面，选中第一个方框中所包含的勾选项，并在其下方配置自定义[安装路径](https://zhida.zhihu.com/search?content_id=533334331&content_type=Answer&match_order=1&q=%E5%AE%89%E8%A3%85%E8%B7%AF%E5%BE%84&zhida_source=entity)；这个路径建议大家自己修改一下，同时记下来这个路径，之后会经常用到。

**注意debug版本**

![https://pic1.zhimg.com/80/v2-bb2cc5e3c8c2055da3e362da635760a7_1440w.webp?source=1def8aca](https://pic1.zhimg.com/80/v2-bb2cc5e3c8c2055da3e362da635760a7_1440w.webp?source=1def8aca)

**依赖库安装**

接下来，我们需要对新创建的**Python**进行`matplotlib`库与`numpy`库的安装。这里就使用**Python**最传统的`pip`安装方法即可，首先输入如下的代码。

**先安装特定版本的numpy**

由于numpy 2.0之后不包含core/include文件夹，目前不清楚在C++中调用方法，因此numpy版本要小于2.0，同时由于1.25.0之后的numpy不支持scipy==1.8.1，选择1.25.0之前的版本

```cpp
pip install numpy==1.24.3
pip install scipy==1.8.1
//缺哪些依赖包就安装哪些依赖，注意版本
```

接下来安装matplotlib：

`pip install -U matplotlib`

出现如下所示的界面即说明`matplotlib`库已经安装完毕。

![https://pic1.zhimg.com/80/v2-9c0c45ca27c972654fad2aaeebc251dd_1440w.webp?source=1def8aca](https://pic1.zhimg.com/80/v2-9c0c45ca27c972654fad2aaeebc251dd_1440w.webp?source=1def8aca)

**4. 解决方案配置**

接下来，我们创建或打开需要调用`matplotlibcpp`库的解决方案。

首先，将前述**Python**安装路径下的以下两个`.dll`文件复制（具体文件名称与**Python**版本有关）。

> **注意：如果没有_d.dll，说明python版本不是debug版，重新安装debug版，或者尝试修改为release版（matplotlib使用release没成功，貌似opencv可以）**
> 

![https://picx.zhimg.com/80/v2-48ce69e14a86984a99a0565300bf9731_1440w.webp?source=1def8aca](https://picx.zhimg.com/80/v2-48ce69e14a86984a99a0565300bf9731_1440w.webp?source=1def8aca)

并将其复制到解决方案的文件夹下。

![https://picx.zhimg.com/80/v2-c3b942d0159c1fc75ac95737b51eed59_1440w.webp?source=1def8aca](https://picx.zhimg.com/80/v2-c3b942d0159c1fc75ac95737b51eed59_1440w.webp?source=1def8aca)

随后，依据文章[疯狂学习GIS：Visual Studio新建项目调用C++已编译的第三方库的方法](https://zhuanlan.zhihu.com/p/573160698)中提到的方法，分别进行以下配置。

首先，在“**附加包含目录**”中，将**Python**和`numpy`库的`include`文件夹放入其中。

![https://picx.zhimg.com/80/v2-239d671ffb49558005b7f356c13f854d_1440w.webp?source=1def8aca](https://picx.zhimg.com/80/v2-239d671ffb49558005b7f356c13f854d_1440w.webp?source=1def8aca)

其次，在“**附加库目录**”中，将**Python**安装路径下`libs`文件夹的路径放入其中。

![https://pica.zhimg.com/80/v2-4a7ed6e8c9c1b3d6a8b2e9433d7077c6_1440w.webp?source=1def8aca](https://pica.zhimg.com/80/v2-4a7ed6e8c9c1b3d6a8b2e9433d7077c6_1440w.webp?source=1def8aca)

再次，在“**附加依赖项**”中，将**Python**安装路径下`libs`文件夹中如下所示的4个`.lib`文件放入其中。

![https://picx.zhimg.com/80/v2-f5891f5a125e8cd50443eed20cdf0226_1440w.webp?source=1def8aca](https://picx.zhimg.com/80/v2-f5891f5a125e8cd50443eed20cdf0226_1440w.webp?source=1def8aca)

随后，对于需要调用`matplotlibcpp`库的程序，需要添加以下代码。

```cpp
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
```

具体如下图所示。

![https://pic1.zhimg.com/80/v2-1e8cfce2c09a945c009984004ba4c6e2_1440w.webp?source=1def8aca](https://pic1.zhimg.com/80/v2-1e8cfce2c09a945c009984004ba4c6e2_1440w.webp?source=1def8aca)

随后，即可开始运行代码。这里提供一个最简单的`matplotlibcpp`库调用代码。

```cpp
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main() {
    plt::plot({ 1, 2, 3, 4 });
    plt::show();
    return 0;
}
```

运行代码，出现如下所示的窗口。

![https://pica.zhimg.com/80/v2-402800519361a404edaccef9e7b92356_1440w.webp?source=1def8aca](https://pica.zhimg.com/80/v2-402800519361a404edaccef9e7b92356_1440w.webp?source=1def8aca)

以上，即完成了`matplotlibcpp`库的配置。

**之后将用到的其他库利用vcpkg安装上，主要是SFML/Graphics.hpp**

### 3. 强化学习第一次作业

如果使用自建的解决方案，将cpp文件导入源文件，将h文件导入头文件

1. 求解贝尔曼方程

使用`第一次作业c++源代码.zip`中的boe.cpp，因为解决方案中的boe.cpp中仅有迭代求解，缺少封闭求解。

2. 求解贝尔曼最优方程

使用`第一次作业c++源代码.zip`中bellmanEquation.cpp代码。

### 4. 强化学习第二次作业

1. 贝尔曼最优方程

使用boe.cpp中的value_iteration（值迭代）、policy_iteration（策略迭代）、truncated_policy_iteration（截断式策略迭代）

具体过程参考main.cpp部分。

对于绘制每个迭代的errors，参考truncated_policy_iteration代码中记录errors和绘制的代码，修改另外两个函数（值迭代、策略迭代），完成绘制。

2. mc epsilon贪心策略

在main中设置参数（最上面），绘制图像（draw_picture.cpp中的drawPoint和drawPoint_million函数），注意使用的基础策略可能不一样，自行设置。

使用mc_sigma.cpp中的mc_epsilon_greedy函数执行策略。该策略对参数设置敏感。

### 5. 强化学习第三次作业

1. 随即梯度下降

该部分与其他部分独立，因此不使用main.cpp，先把里面的main函数注释掉。

使用gradient_descend.cpp中函数完成题目。

2. Q-learning(off-policy)

使用前先把gradient_descend.cpp中main函数注释掉。

### 6.  强化学习第四次作业

第四次作业：
1. 完成TD-Linear：td_linear.cpp、draw_3d.py
2. 完成Deep Q-learning：dqn.cpp
3. 完成nn.cpp、nn.h

说明：
1. td_linear.cpp中内容可以在main.cpp中调用，生成的V值记录下来，通过python代码(draw_3d.py)绘制作业要求的3D图像。
2. dqn.cpp由于没有整理头文件，直接调用dqn.cpp中的int main()运行。
3. 注意同一个项目中只能有一个int main()，因此在使用td_linear或者dqn时，需要把对方对应的int main()注释掉。
4. nn.cpp为手动实现的具有单隐层的神经网络，包括前向传播与后像传播，目前泛用性一般。


