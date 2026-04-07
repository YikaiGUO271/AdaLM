"""
测试函数模型集合
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, hessian

class HighDimRosenbrock:
    """高维Rosenbrock函数"""
    def __init__(self, dim=200, a=1, b=100):
        self.dim = dim
        self.a = a
        self.b = b
        
        self._loss_jitted = jax.jit(self._loss)
        self._gradient_jitted = jax.jit(jax.grad(self._loss))
        self._hessian_jitted = jax.jit(jax.hessian(self._loss))
    
    def _loss(self, x):
        terms = self.b * (x[1:] - x[:-1]**2)**2 + (self.a - x[:-1])**2
        return jnp.sum(terms)
    
    def loss(self, x):
        return self._loss_jitted(x).item()
    
    def gradient(self, x):
        return self._gradient_jitted(x)
    
    def hessian(self, x):
        return self._hessian_jitted(x)


class PolytopeFeasibility:
    """多面体可行性问题"""
    def __init__(self, dim=100, m=1000, p=2):
        self.dim = dim
        self.m = m
        self.p = p
        self.A = jnp.array(np.random.randn(m, dim))
        self.b = jnp.array(np.random.randn(m))
        
        self._loss_jitted = jax.jit(self._loss)
        self._gradient_jitted = jax.jit(jax.grad(self._loss))
        self._hessian_jitted = jax.jit(jax.hessian(self._loss))
    
    def _loss(self, x):
        residuals = self.A @ x - self.b
        positive_residuals = jnp.maximum(0, residuals)
        return jnp.sum(positive_residuals ** self.p)
    
    def loss(self, x):
        return self._loss_jitted(x).item()
    
    def gradient(self, x):
        return self._gradient_jitted(x)
    
    def hessian(self, x):
        return self._hessian_jitted(x)


class WorstInstancesFunction:
    """Worst instances函数"""
    def __init__(self, dim=200, q=3):
        self.dim = dim
        self.q = q
        
        self._loss_jitted = jax.jit(self._loss)
        self._gradient_jitted = jax.jit(jax.grad(self._loss))
        self._hessian_jitted = jax.jit(jax.hessian(self._loss))
    
    def _loss(self, x):
        diff_terms = jnp.sum(jnp.abs(x[1:] - x[:-1])**self.q)
        last_term = jnp.abs(x[-1])**self.q
        return (diff_terms + last_term) / self.q
    
    def loss(self, x):
        return self._loss_jitted(x).item()
    
    def gradient(self, x):
        return self._gradient_jitted(x)
    
    def hessian(self, x):
        return self._hessian_jitted(x)


class ZakharovFunction:
    """Zakharov函数"""
    def __init__(self, dim=200):
        self.dim = dim
        self.i_vec = 0.5 * jnp.arange(1, dim + 1)
        
        self._loss_jitted = jax.jit(self._loss)
        self._gradient_jitted = jax.jit(jax.grad(self._loss))
        self._hessian_jitted = jax.jit(jax.hessian(self._loss))
    
    def _loss(self, x):
        linear_term = jnp.sum(self.i_vec * x)
        return jnp.sum(x**2) + linear_term**2 + linear_term**4
    
    def loss(self, x):
        return self._loss_jitted(x).item()
    
    def gradient(self, x):
        return self._gradient_jitted(x)
    
    def hessian(self, x):
        return self._hessian_jitted(x)


class PowellSingularFunction:
    """Powell奇异函数"""
    def __init__(self, dim=200):
        self.dim = dim
        assert dim % 4 == 0, "Dimension must be divisible by 4"
        self.groups = dim // 4
        
        self._loss_jitted = jax.jit(self._loss)
        self._gradient_jitted = jax.jit(jax.grad(self._loss))
        self._hessian_jitted = jax.jit(jax.hessian(self._loss))
    
    def _loss(self, x):
        total = 0.0
        for i in range(self.groups):
            x1 = x[4*i]
            x2 = x[4*i+1]
            x3 = x[4*i+2]
            x4 = x[4*i+3]
            
            term1 = (x1 + 10*x2) ** 2
            term2 = 5 * (x3 - x4) ** 2
            term3 = (x2 - 2*x3) ** 4
            term4 = 10 * (x1 - x4) ** 4
            
            total += term1 + term2 + term3 + term4
        return total
    
    def loss(self, x):
        return self._loss_jitted(x).item()
    
    def gradient(self, x):
        return self._gradient_jitted(x)
    
    def hessian(self, x):
        return self._hessian_jitted(x)


class LogSumExpFunction:
    """Log-Sum-Exp测试函数"""
    def __init__(self, n=500, d=200, rho=0.5, random_state=42):
        self.n = n
        self.d = d
        self.rho = rho
        
        key = jax.random.PRNGKey(random_state)
        key_A, key_b = jax.random.split(key)
        self.A = jax.random.normal(key_A, (n, d))
        self.b = jax.random.normal(key_b, (n,))
        
        self._loss_jitted = jax.jit(self._loss)
        self._gradient_jitted = jax.jit(jax.grad(self._loss))
        self._hessian_jitted = jax.jit(jax.hessian(self._loss))
    
    def _loss(self, x):
        linear_terms = jnp.dot(self.A, x) - self.b
        max_term = jnp.max(linear_terms)
        shifted_terms = (linear_terms - max_term) / self.rho
        exp_terms = jnp.exp(shifted_terms)
        return jnp.log(jnp.sum(exp_terms)) + max_term
    
    def loss(self, x):
        return float(self._loss_jitted(x))
    
    def gradient(self, x):
        return self._gradient_jitted(x)
    
    def hessian(self, x):
        return self._hessian_jitted(x)


class MultivariateTMLE:
    """多元t分布最大似然估计（无约束参数化版本）"""
    
    def __init__(self, n_samples=1000, dim=10, df_true=5.0, random_state=42):
        """
        Args:
            n_samples: 样本数量
            dim: 数据维度
            df_true: 真实的自由度参数
            random_state: 随机种子
        """
        self.n_samples = n_samples
        self.dim = dim
        self.df_true = df_true
        
        # 生成模拟数据
        key = jax.random.PRNGKey(random_state)
        key1, key2, key3 = jax.random.split(key, 3)
        
        # 生成真实的参数
        self.mu_true = jax.random.normal(key1, (dim,)) * 1.0  # 真实均值
        L_true = jax.random.normal(key2, (dim, dim)) * 1
        self.Sigma_true = L_true @ L_true.T + jnp.eye(dim) * 1  # 真实协方差矩阵
        
        # 生成多元t分布数据
        self.X = self._generate_multivariate_t(
            key3, self.mu_true, self.Sigma_true, df_true, n_samples
        )
        
        # 预计算一些常量
        self.n_L_params = self.dim * (self.dim + 1) // 2
        self.total_params = self.dim + self.n_L_params + 1  # mu + L + nu_tilde
        
        # 预计算索引映射（在初始化时计算，避免在JIT中使用Python控制流）
        self.L_indices = self._precompute_L_indices()
        
        # 使用JAX编译关键函数
        self._loss_jitted = jit(self._loss)
        self._gradient_jitted = jit(grad(self._loss))
        self._hessian_jitted = jit(hessian(self._loss))
        
    
    def _precompute_L_indices(self):
        """预计算L矩阵的索引映射"""
        indices = []
        idx = 0
        for i in range(self.dim):
            for j in range(i + 1):
                indices.append((i, j, idx))
                idx += 1
        return indices
    
    def _generate_multivariate_t(self, key, mu, Sigma, df, n_samples):
        """生成多元t分布样本"""
        key_gamma, key_normal = jax.random.split(key)
        
        # 1. 从Gamma分布生成缩放因子
        shape = df / 2.0
        rate = df / 2.0
        u = jax.random.gamma(key_gamma, a=shape, shape=(n_samples,)) / rate
        
        # 2. 从多元正态生成基础样本
        L = jnp.linalg.cholesky(Sigma)
        z = jax.random.normal(key_normal, (n_samples, self.dim))
        normal_samples = mu + jnp.dot(z, L.T)
        
        # 3. 应用缩放得到t分布样本
        scaling = jnp.sqrt(1.0 / u)[:, jnp.newaxis]
        t_samples = normal_samples * scaling
        
        return t_samples
    
    def _unpack_parameters(self, theta):
        """从无约束参数向量解包参数"""
        # 参数顺序: [mu, L_flat, nu_tilde]
        mu = theta[:self.dim]
        L_flat = theta[self.dim:self.dim + self.n_L_params]
        nu_tilde = theta[self.dim + self.n_L_params]
        
        return mu, L_flat, nu_tilde
    
    def _reconstruct_L(self, L_flat):
        """从扁平化参数重建下三角矩阵L - 完全JAX兼容版本"""
        L = jnp.zeros((self.dim, self.dim))
        
        # 使用静态的索引映射
        for i, j, flat_idx in self.L_indices:
            value = L_flat[flat_idx]
            if i == j:
                value = jnp.exp(value)  # 对角线取指数确保正定
            L = L.at[i, j].set(value)
        
        return L
    
    def _loss(self, theta):
        """无约束负对数似然函数 - 完全向量化版本"""
        mu, L_flat, nu_tilde = self._unpack_parameters(theta)
        
        # 重建参数
        L = self._reconstruct_L(L_flat)
        Sigma = L @ L.T  # 尺度矩阵
        nu = jnp.exp(nu_tilde)  # 自由度
        
        # 计算常数项
        log_gamma_term = (jax.scipy.special.gammaln((nu + self.dim) / 2) - 
                         jax.scipy.special.gammaln(nu / 2))
        log_det_term = 0.5 * jnp.linalg.slogdet(Sigma)[1]
        log_const = log_gamma_term - 0.5 * self.dim * jnp.log(nu * jnp.pi) - log_det_term
        
        # 向量化计算所有样本的马氏距离
        diff = self.X - mu  # (n_samples, dim)
        
        # 使用solve_triangular批量计算
        L_inv_diff = jax.vmap(lambda x: jax.scipy.linalg.solve_triangular(L, x, lower=True))(diff)
        mahalanobis_sq = jnp.sum(L_inv_diff ** 2, axis=1)  # (n_samples,)
        
        # 向量化计算数据项
        data_terms = (nu + self.dim) / 2 * jnp.log(1 + mahalanobis_sq / nu)
        total_data_term = jnp.sum(data_terms)
        
        # 负对数似然
        nll = -self.n_samples * log_const + total_data_term
        
        # 添加小的正则化项防止数值问题
        reg_term = 0*1e-8 * (jnp.sum(mu ** 2) + jnp.sum(L_flat ** 2) + nu_tilde ** 2)
        
        return nll + reg_term
    
    def loss(self, theta):
        """计算损失（负对数似然）"""
        # 确保输入是JAX数组
        theta = jnp.asarray(theta)
        return float(self._loss_jitted(theta))
    
    def gradient(self, theta):
        """计算梯度"""
        theta = jnp.asarray(theta)
        grad_val = self._gradient_jitted(theta)
        return np.array(grad_val)  # 转换为numpy数组以便与其他代码兼容
    
    def hessian(self, theta):
        """计算Hessian矩阵"""
        theta = jnp.asarray(theta)
        hess_val = self._hessian_jitted(theta)
        return np.array(hess_val)  # 转换为numpy数组
    
    def get_initial_guess(self):
        """获取合理的初始参数猜测"""
        # 使用样本矩作为初始猜测
        sample_mean = jnp.mean(self.X, axis=0)
        sample_cov = jnp.cov(self.X, rowvar=False)
        
        # 对样本协方差进行正则化确保正定
        sample_cov_reg = sample_cov + jnp.eye(self.dim) * 0.00
        
        # 计算初始Cholesky分解
        try:
            L0 = jnp.linalg.cholesky(sample_cov_reg)
        except:
            # 如果Cholesky失败，使用单位矩阵
            L0 = jnp.eye(self.dim)
        
        # 扁平化L矩阵（对角线取对数）
        L_flat = []
        for i in range(self.dim):
            for j in range(i + 1):
                if i == j:
                    L_flat.append(float(jnp.log(jnp.maximum(L0[i, j], 1e-8))))
                else:
                    L_flat.append(float(L0[i, j]))
        L_flat = np.array(L_flat)
        
        # 初始自由度猜测
        nu_tilde_0 = np.log(10.0)  # 对应 nu=10
        
        # 组合所有参数
        theta0 = np.concatenate([np.array(sample_mean), L_flat, np.array([nu_tilde_0])])
        
        return theta0  # 返回numpy数组
    
    def get_parameter_count(self):
        """返回参数总数"""
        return self.total_params
    
    def evaluate_estimation(self, theta):
        """评估参数估计质量"""
        theta = jnp.asarray(theta)
        mu_est, L_flat_est, nu_tilde_est = self._unpack_parameters(theta)
        nu_est = jnp.exp(nu_tilde_est)
        L_est = self._reconstruct_L(L_flat_est)
        Sigma_est = L_est @ L_est.T
        d=self.dim
        mu_error = float(jnp.linalg.norm(mu_est - self.mu_true)/d)
        Sigma_error = float(jnp.linalg.norm(Sigma_est - self.Sigma_true)/((1+d)*d/2))
        nu_error = float(jnp.abs(nu_est - self.df_true))
        
        return {
            'mu_error': mu_error,
            'Sigma_error': Sigma_error,
            'nu_error': nu_error,
            'mu_estimated': np.array(mu_est),
            'Sigma_estimated': np.array(Sigma_est),
            'nu_estimated': float(nu_est)
        }