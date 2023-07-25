## Problem Set 2

### 1 Regularized Normal Equation for Linear Regression

> Given a data set $\{x^{(i)},y^{(i)}\}_{i=1,\dots m}$ with $x^{(i)}\in \mathbb{R}^n$ and $y^{(i)}\in\mathbb{R}$, the general form of regularized linear regression is as follows
> $$
> \mathop{min}_\theta\frac{1}{2m}\big[ \sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})^2+\lambda\sum^n_{j=1}\theta^2_j\big]
> $$
> Derive the normal equation.

对于正则化线性回归的代价函数：
$$
J(\theta)=\frac{1}{2m}\big[ \sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})^2+\lambda\sum^n_{j=1}\theta^2_j\big]
$$
代入假设函数：$h_\theta(x^{(i)})=x^{(i)}\theta$，转换为矩阵形式，得到：
$$
\begin{align*}
J(\theta)
& = \frac{1}{2}\big[(X\theta-Y)^T(X\theta-Y)+\lambda A^TA  \big] \\
& = \frac{1}{2}\big[(\theta^TX^T-Y^T)(X\theta-Y)+\lambda A^TA  \big] \\
& = \frac{1}{2}(\theta^TX^TX\theta-\theta^TX^TY-Y^TX\theta+Y^TY+\lambda A^TA)
\end{align*}
$$
其中：
$$
L=
\begin{bmatrix}
0 \\
& 1 \\
& & \ddots \\
& & & 1 
\end{bmatrix}
,\quad
A=L\theta=
\begin{bmatrix}
0 \\
\theta_1 \\
\vdots \\
\theta_n 
\end{bmatrix}
\quad
$$
对$J(\theta)$求关于$\theta$的偏导：
$$
\begin{align}
\frac{\partial}{\partial\theta}J(\theta)
& = \frac{1}{2}\big[2X^TX\theta-X^TY-(Y^TX)^T+0+2\lambda A  \big] \\
& = \frac{1}{2}(2X^TX\theta-2X^TY+2\lambda A) \\
& = X^TX\theta-X^TY+\lambda A \\
\end{align}
$$

令$\frac{\partial}{\partial\theta}J(\theta)=0$，得：
$$
X^TX\theta+\lambda A = X^TY
$$
即：
$$
X^TX\theta+\lambda L\theta = X^TY
$$

$$
(X^TX+\lambda L)\theta = X^TY
$$

等号两侧左乘$(X^TX+\lambda L)^{-1}$，得：
$$
\theta=(X^TX+\lambda L)^{-1}X^TY
$$
其中：
$$
L=
\begin{bmatrix}
0 \\
& 1 \\
& & \ddots \\
& & & 1 
\end{bmatrix}
$$



### 2 Gaussian Discriminant Analysis Model

> Given m training data $\{x^{(i)},y^{(i)}\}_{i=1,\dots m}$, assume that $y\sim Bernoulli(\psi)$， $x|y=0\sim \mathcal{N}(μ_0, \Sigma)$， $x|y=1\sim\mathcal{N}(μ_1,\Sigma)$. Hence, we have
> $$
> \bullet \quad p(y) = \psi^y(1-\psi)^{1-y} \hfill \\
> \bullet \quad p(x|y=0) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}exp(-\frac{1}{2}(x-μ_0)^T\Sigma^{-1}(x-μ_0))\hfill \\ 
> \bullet \quad p(x|y=1) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}exp(-\frac{1}{2}(x-μ_1)^T\Sigma^{-1}(x-μ_1)) \hfill
> $$
> The log-likelihood function is
> $$
> \begin{align*}
> \ell(\psi, μ_0, μ_1, \Sigma)
> &=log\prod^m_{i=1}p(x^{(i)},y^{(i)};\psi,μ_0, μ_1, \Sigma) \\
> &=log\prod^m_{i=1}p(x^{(i)}|y^{(i)}; \psi,μ_0,μ_1, \Sigma)p(y^{(i)};\psi)
> \end{align*}
> $$
> Solve $\psi$, $μ_0$, $μ_1$ and $\Sigma$ by maximizing $\ell(\psi, μ_0, μ_1, \Sigma)$.
>
> Hint: $\nabla_Xtr(AX^{-1}B)=-(X^{-1}BAX^{-1})^T$, $\nabla_A|A|=|A|(A^{-1})^T$

由题意知：
$$
\begin{align*}
\ell(\psi, μ_0, μ_1, \Sigma)
&=log\prod^m_{i=1}p(x^{(i)}|y^{(i)}; \psi,μ_0,μ_1, \Sigma)p(y^{(i)};\psi) \\
&=\sum^m_{i=1}(log\ p(x^{(i)}|y^{(i)};\mu_0,\mu_1,\Sigma)+log\ p(y^{(i)};\psi)) \\
&=\sum^m_{i=1}\big[log\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}+(-\frac{1}{2}(x^{(i)}-\mu_{y^{(i)}})^T\Sigma^{-1}(x^{(i)}-\mu_{y^{(i)}}))\\
&\quad+y^{(i)}log\ \psi+(1-y^{(i)})log(1-\psi)  \big] \\
&=\sum^m_{i=1}\big[-\frac{n}{2}log\ 2\pi-\frac{1}{2}log|\Sigma|-\frac{1}{2}(x^{(i)}-\mu_{y^{(i)}})^T\Sigma^{-1}(x^{(i)}-\mu_{y^{(i)}})\\
&\quad+y^{(i)}log\ \psi+(1-y^{(i)})log(1-\psi)\big]
\end{align*}
$$
对$\ell(\psi,\mu_0,\mu_1,\Sigma)$关于$\psi$求偏导：
$$
\begin{align}
\frac{\partial\ell(\psi,\mu_0,\mu_1,\Sigma)}{\partial\psi}
&=\sum^m_{i=1}(\frac{y^{(i)}}{\psi}-\frac{1-y^{(i)}}{1-\psi})
\end{align}
$$
令该式等于0，有：
$$
\begin{align}
&\quad\sum^m_{i=1}(\frac{y^{(i)}}{\psi}-\frac{1-y^{(i)}}{1-\psi})=0 \\
\Rightarrow&\quad\sum^m_{i=1}[y^{(i)}(1-\psi)-(1-y^{(i)})\psi]=0 \\
\Rightarrow&\quad\sum^m_{i=1}(y^{(i)}-\psi)=0\\
\Rightarrow&\quad m\psi=\sum^m_{i=1}y^{(i)}\\
\end{align}
$$
得：
$$
\psi=\frac{1}{m}\sum^m_{i=1}1\{y^{(i)}=1\}
$$
令$x\in\mathbb{R}^{n×1}$，$A\in\mathbb{R}^{n×n}$，对于$\frac{\part x^TAx}{\part x}$，有：
$$
\begin{align}
\frac{\part x^TAx}{\part x}
&=	{
	\begin{bmatrix}
	\frac{\part x^TAx}{\part x_1}\\
	\frac{\part x^TAx}{\part x_2}\\
	\vdots\\
	\frac{\part x^TAx}{\part x_n}
	\end{bmatrix}
	}\\
&=	{
	\begin{bmatrix}
	\frac{\part\sum^n_{i=1}\sum^n_{j=1}x_iA_{ij}x_j}{\part x_1}\\
	\frac{\part\sum^n_{i=1}\sum^n_{j=1}x_iA_{ij}x_j}{\part x_2}\\
	\vdots\\
	\frac{\part\sum^n_{i=1}\sum^n_{j=1}x_iA_{ij}x_j}{\part x_n}
	\end{bmatrix}
	}\\
&=	{
	\begin{bmatrix}
	\sum^n_{i=1}A_{i1}x_i+\sum^n_{j=1}A_{1j}x_j\\
	\sum^n_{i=1}A_{i2}x_i+\sum^n_{j=1}A_{2j}x_j\\
	\vdots\\
	\sum^n_{i=1}A_{in}x_i+\sum^n_{j=1}A_{nj}x_j
	\end{bmatrix}
	}\\
&= 	{
	\begin{bmatrix}
	\sum^n_{i=1}A_{i1}x_i\\
	\sum^n_{i=1}A_{i2}x_i\\
	\vdots\\
	\sum^n_{i=1}A_{in}x_i
	\end{bmatrix}
	}
+  	{
	\begin{bmatrix}
	\sum^n_{j=1}A_{1j}x_j\\
	\sum^n_{j=1}A_{2j}x_j\\
	\vdots\\
	\sum^n_{j=1}A_{nj}x_j
	\end{bmatrix}
	}\\
&=Ax+A^Tx
\end{align}
$$
当$A$为对称矩阵时，有：$A=A^T$，因此：
$$
\frac{\part x^TAx}{\part x}=Ax+A^Tx=2Ax
$$


而对于对称的协方差矩阵$\Sigma$，显然满足该条件，在上式的基础上，对$\ell(\psi,\mu_0,\mu_1,\Sigma)$关于$\mu_0$求偏导：
$$
\begin{align}
\frac{\partial\ell(\psi,\mu_0,\mu_1,\Sigma)}{\partial\mu_0}
&=\sum^m_{i=1}(\frac{1}{2}·2·\Sigma^{-1}(x^{(i)}-\mu_0)·1\{y^{(i)}=0\})\\
&=\sum^m_{i=1}\Sigma^{-1}(x^{(i)}-\mu_0)·1\{y^{(i)}=0\}
\end{align}
$$
令上式为0：
$$
\sum^m_{i=1}\Sigma^{-1}(x^{(i)}-\mu_0)·1\{y^{(i)}=0\}=0\\
$$
$\Sigma$为协方差矩阵，故$\Sigma^{-1}$不为0，可约去，有：
$$
\sum^m_{i=1}\mu_0·1\{y^{(i)}=0\}=\sum^m_{i=1}x^{(i)}·1\{y^{(i)}=0\}
$$
得：
$$
\mu_0=\frac{\sum^m_{i=1}x^{(i)}·1\{y^{(i)}=0\}}{\sum^m_{i=1}·1\{y^{(i)}=0\}}
$$
同理可得：
$$
\mu_1=\frac{\sum^m_{i=1}x^{(i)}·1\{y^{(i)}=1\}}{\sum^m_{i=1}·1\{y^{(i)}=1\}}
$$
由题目中提供的公式$\nabla_Xtr(AX^{-1}B)=-(X^{-1}BAX^{-1})^T$, $\nabla_A|A|=|A|(A^{-1})^T$，

对$\ell(\psi,\mu_0,\mu_1,\Sigma)$关于$\Sigma$求偏导：
$$
\begin{align}
\frac{\partial\ell(\psi,\mu_0,\mu_1,\Sigma)}{\partial\Sigma}
&=\sum^m_{i=1}\big[-\frac{1}{2}\frac{1}{|\Sigma|}·|\Sigma|·(\Sigma^{-1})^T+\frac{1}{2}(\Sigma^{-1}(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T\Sigma^{-1})^T \big]\\
&=\sum^m_{i=1}\big[-\frac{1}{2}(\Sigma^{-1})^T+\frac{1}{2}(\Sigma^{-1})^T(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T(\Sigma^{-1})^T \big]\\
\end{align}
$$
由于$\Sigma$是对称的，有：
$$
\begin{align}
\frac{\partial\ell(\psi,\mu_0,\mu_1,\Sigma)}{\partial\Sigma}
&=\sum^m_{i=1}\big[-\frac{1}{2}(\Sigma^{T})^{-1}+\frac{1}{2}(\Sigma^{T})^{-1}(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T(\Sigma^{T})^{-1} \big]\\
&=\sum^m_{i=1}\big[-\frac{1}{2}\Sigma^{-1}+\frac{1}{2}\Sigma^{-1}(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T\Sigma^{-1} \big]
\end{align}
$$
令该式等于0，有：
$$
\begin{align}
&\quad\sum^m_{i=1}\big[-\frac{1}{2}\Sigma^{-1}+\frac{1}{2}\Sigma^{-1}(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T\Sigma^{-1} \big]=0\\
\Rightarrow&\quad\sum^m_{i=1}\big[1-\Sigma^{-1}(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T\big]=0\\
\Rightarrow&\quad m=\sum^m_{i=1}\Sigma^{-1}(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T\\
\Rightarrow&\quad m\Sigma=\sum^m_{i=1}(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T
\end{align}
$$
得：
$$
\Sigma=\frac{1}{m}\sum^m_{i=1}(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T
$$




### 3 MLE for Naive Bayes

> Consider the following definition of **MLE problem for multinomials**. The input to the problem is a finite set $\mathcal{Y}$, and a weight $c_y \ge 0$ for each $y\in\mathcal{Y}$.
>
> The output from the problem is the distribution $p^*$ that solves the following maximization problem.
> $$
> p^*=arg\ \mathop{max}_{p\in \mathcal{P_\mathcal{Y}}}\sum_{y\in \mathcal{Y}}c_ylog\ (p_y)
> $$

>(i) Prove that, the vector $p^*$ has components
>$$
>p^*_y=\frac{c_y}{N}
>$$
>for $\forall y\in\mathcal{Y}$, where $N=\sum_{y\in\mathcal{Y}}c_y$.
>
>Hint: Use the theory of Lagrange multiplier.

要求解该问题：
$$
\mathop{max}_{p\in\mathcal{P_\mathcal{y}}}\sum_{y\in\mathcal{Y}}c_ylog\ p_y
$$
即求解：
$$
\begin{align}
&\quad\mathop{min}_{p\in\mathcal{P_\mathcal{y}}}(-\sum_{y\in\mathcal{Y}}c_ylog\ p_y)\\
s.t.&\quad\sum_{y\in\mathcal{Y}}p_y=1,\quad p_y\ge0
\end{align}
$$
由上述条件构建拉格朗日问题：
$$
L(c_y,p_y)=-\sum_{y\in\mathcal{Y}}c_ylog\ p_y+\lambda(\sum_{y\in\mathcal{Y}}p_y-1)-\sum_{y\in\mathcal{Y}}\mu_yp_y
$$
其中$\mu_y\ge0$。

关于$\forall y\in\mathcal{Y}$求偏导，均有：
$$
\frac{\part L(c_y, p_y)}{\part p_y}=-\frac{c_y}{p_y}+\lambda-\mu_y
$$
令该式等于0，得：
$$
c_y=\lambda p_y-\mu_yp_y
$$
由拉格朗日问题性质，有：
$$
\mu_yp_y=0,\quad \forall y\in\mathcal{Y}
$$
代入上式，有：
$$
p_y=\frac{c_y}{\lambda}
$$
由：
$$
\begin{align}
&\quad \sum_{y\in\mathcal{Y}}p_y=1 \\
\Rightarrow &\quad \sum_{y\in\mathcal{Y}}\frac{c_y}{\lambda}=1\\
\Rightarrow &\quad \lambda=\sum_{y\in\mathcal{Y}}c_y
\end{align}
$$
代入上式，得证：
$$
p_y=\frac{c_y}{\sum_{y\in\mathcal{Y}}c_y}
$$




> (ii) Using the above consequence, prove that, the maximum-likelihood estimates for Naive Bayes model are as follows
> $$
> p(y)=\frac{\sum^m_{i=1}1(y^{(i)}=y)}{m}
> $$
> and
> $$
> p_j(x|y)=\frac{\sum^m_{i=1}1(y^{(i)}=y\and x^{(i)}_j=x)}{\sum^m_{i=1}1(y^{(i)}=y)}
> $$

对于对数似然函数：
$$
\begin{align}
\ell(\Omega)
&=log\prod^m_{i=1}p(x^{(i)},y^{(i)})\\
&=\sum^m_{i=1}log\ p(x^{(i)}, y^{(i)})\\
&=\sum^m_{i=1}log\ (p(y^{(i)})\prod^n_{j=1}p_j(x^{(i)}_j|y^{(i)}))\\
&=\sum^m_{i=1}log\ p(y^{(i)})+\sum^m_{i=1}\sum^n_{j=1}log\ p_j(x^{(i)}_j|y^{(i)})\\
&=\sum_{y\in\mathcal{Y}}count(y)log\ p_y+\sum^n_{j=1}\sum_{y\in\mathcal{Y}}\sum_{x\in\{-1,+1\}}count_j(x|y)log\ p_j(x|y)\\
\end{align}
$$
其中：
$$
\begin{align}
&count(y)=\sum^m_{i=1}1\{y^{(i)}=y\}\\
&count_j(x|y)=\sum^m_{i=1}1\{y^{(i)}=y\and x^{(i)}_j=x\}
\end{align}
$$
要最大化上式，可分别最大化上式中的两部分。

对于对数似然函数中的第一部分：
$$
\begin{align}
&\quad\mathop{max}\sum_{y\in\mathcal{Y}}count(y)log\ p_y\\
s.t.&\quad\sum_{y\in\mathcal{Y}}p_y=1,\quad p_y\ge0
\end{align}
$$
等价于(i)中的问题，可得：
$$
p_y=\frac{count(y)}{\sum_{y\in\mathcal{Y}}count(y)}=\frac{\sum^m_{i=1}1\{y^{(i)}=y\}}{m}
$$
同理，对于第二部分：
$$
\begin{align}
&\quad\mathop{max}\sum^n_{j=1}\sum_{y\in\mathcal{Y}}\sum_{x\in\{-1,+1\}}count_j(x|y)log\ p_j(x|y)\\
s.t.&\quad\sum^n_{j=1}\sum_{y\in\mathcal{Y}}\sum_{x\in\{-1,+1\}}p_j(x|y)=1,\quad p_j(x|y)\ge0
\end{align}
$$
可得：
$$
p_j(x|y)=\frac{count_j(x|y)}{\sum^n_{j=1}\sum_{y\in\mathcal{Y}}\sum_{x\in\{-1,+1\}}count_j(x|y)}=\frac{\sum^m_{i=1}1(y^{(i)}=y\and x^{(i)}_j=x)}{\sum^m_{i=1}1(y^{(i)}=y)}
$$
