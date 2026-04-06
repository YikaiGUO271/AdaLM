# 优化器实现
class AdaN(BaseOptimizer):
    """自适应牛顿法 (Adaptive Newton)"""
    
    def __init__(self, H0=1.0, max_inner_iter=20):
        """
        Args:
            H0: 初始正则化参数
            max_inner_iter: 最大内循环迭代次数
        """
        super().__init__('AdaN')
        self.H0 = H0
        self.max_inner_iter = max_inner_iter
        self.history_H = []
        self.history_lambda = []
        self.history_inner_iters = []
    
    def optimize(self, model, dim, initial_theta=None, max_iter=100, tol=1e-8, **kwargs):
        self.reset_history()
        self.history_H = []
        self.history_lambda = []
        self.history_inner_iters = []
        
        theta = initial_theta if initial_theta is not None else jnp.array(np.random.randn(dim) * 0 + 1.5)
        start_time = time.time()
        H_k = self.H0
        
        for k in range(max_iter):
            self._record(theta, model, start_time)
            
            grad = model.gradient(theta)
            grad_norm = jnp.linalg.norm(grad)
            
            # 检查收敛
            if grad_norm < tol:
                break
            
            # 内循环：寻找满足条件的正则化参数
            inner_success = False
            inner_iters = 0
            lambda_k = 0.0
            theta_plus = None
            
            # 从减小的H_k开始（第k>0次迭代）
            if k > 0 and len(self.history_H) > 0:
                H_k = max(self.history_H[-1] / 4, self.H0)  # 从之前的H_k/4开始
                hess = model.hessian(theta)
            for n in range(self.max_inner_iter):
                inner_iters = n + 1
                # 计算lambda_k
                lambda_k = jnp.sqrt(H_k * grad_norm)
                
                # 计算阻尼牛顿步
                
                try:
                    A = hess + lambda_k * jnp.eye(dim)
                    L = jax.scipy.linalg.cho_factor(A, lower=True)
                    delta_theta = jax.scipy.linalg.cho_solve(L, -grad)
                    theta_plus = theta + delta_theta
                except:
                    # 如果Cholesky分解失败，增加正则化继续
                    H_k *= 2
                    continue
                
                # 计算步长范数
                r_plus = jnp.linalg.norm(delta_theta)
                
                # 计算新点的梯度
                grad_plus = model.gradient(theta_plus)
                grad_plus_norm = jnp.linalg.norm(grad_plus)
                
                # 计算新点的函数值
                f_plus = model.loss(theta_plus)
                f_current = model.loss(theta)
                
                # 检查停止条件
                condition1 = grad_plus_norm <= 2 * lambda_k * r_plus
                condition2 = f_plus <= f_current - (2/3) * lambda_k * (r_plus ** 2)
                
                if condition1 and condition2:
                    inner_success = True
                    break
                else:
                    # 条件不满足，增加正则化
                    H_k *= 2
            
            # 记录内循环信息
            self.history_H.append(H_k)
            self.history_lambda.append(lambda_k)
            self.history_inner_iters.append(inner_iters)
            
            if inner_success and theta_plus is not None:
                theta = theta_plus
            else:
                # 如果内循环失败，使用梯度下降作为备选
                step_size = 1.0 / (grad_norm + 1e-12)
                theta = theta - step_size * grad
        
        return theta
    
    def get_detailed_history(self):
        """获取详细的优化历史"""
        return {
            'loss': self.history.get('loss', []),
            'grad_norm': self.history.get('grad_norm', []),
            'time': self.history.get('time', []),
            'H_values': self.history_H,
            'lambda_values': self.history_lambda,
            'inner_iterations': self.history_inner_iters
        }

class CR(BaseOptimizer):
    """自适应正则化立方算法（JAX版本）- 严格内循环版"""
    def __init__(self, sigma0=1.0, max_inner_iter=20):
        super().__init__('CR')
        self.sigma = sigma0
        self.max_inner_iter = max_inner_iter
    
    def optimize(self, model, dim, initial_theta=None, max_iter=100, tol=1e-7, **kwargs):
        self.reset_history()
        theta = initial_theta if initial_theta is not None else jnp.array(np.random.randn(dim) * 0 + 1.5)
        start_time = time.time()
        
        iter_count = 0
        inner_iter_total = 0
        
        while iter_count < max_iter:
            self._record(theta, model, start_time)
            grad = model.gradient(theta)
            hess = model.hessian(theta)
            
            if jnp.linalg.norm(grad) < tol:
                break
            
            step_accepted = False
            
            for inner_iter in range(self.max_inner_iter):
                inner_iter_total += 1
                
                try:
                    # 计算最小特征值
                    lambda_min = jnp.linalg.eigvalsh(hess)[0]
                    r_low = jnp.maximum(0, -2*lambda_min/self.sigma) + 1e-8
                    
                    # 定义phi函数
                    def phi(r):
                        A = hess + (self.sigma * r / 2) * jnp.eye(dim)
                        try:
                            # 使用Cholesky分解
                            L = jax.scipy.linalg.cho_factor(A, lower=True)
                            d = -jax.scipy.linalg.cho_solve(L, grad)
                            return jnp.linalg.norm(d) - r
                        except:
                            return jnp.inf
                    
                    # 寻找r的上界
                    r_high = r_low + 1.0
                    while phi(r_high) > 0:
                        r_high *= 2
                        if r_high > 1e12:
                            break
                    
                    # 求解r_opt
                    try:
                        r_opt = brentq(lambda r: phi(r).item(), r_low.item(), r_high.item(), rtol=1e-6, maxiter=100)
                    except:
                        # 如果求解失败，使用中间值
                        r_opt = (r_low + r_high) / 2
                        r_opt = r_opt.item() if hasattr(r_opt, 'item') else r_opt
                        
                    # 计算更新方向
                    A = hess + (self.sigma * r_opt / 2 + 1e-8) * jnp.eye(dim)
                    L = jax.scipy.linalg.cho_factor(A, lower=True)
                    delta_theta = -jax.scipy.linalg.cho_solve(L, grad)
                    
                    # 计算实际下降和预测下降
                    loss_val = model.loss(theta)
                    loss_new = model.loss(theta + delta_theta)
                    actual_reduction = loss_val - loss_new
                    
                    # 预测下降量
                    predicted_reduction = -0.5 * jnp.dot(grad, delta_theta) 
                    predicted_reduction -= 0.5 * jnp.dot(delta_theta, jnp.dot(hess, delta_theta))
                    predicted_reduction -= (self.sigma/6) * jnp.linalg.norm(delta_theta)**3
                    
                    # 避免除零和数值问题
                    rho = actual_reduction - predicted_reduction
                    
                    # 检查是否接受这一步
                    if rho >= 0.0 and actual_reduction > 0:  # 成功迭代
                        theta = theta + delta_theta
                        self.sigma = max(self.sigma * 0.5, 1e-8)  # 减小sigma
                        step_accepted = True
                        iter_count += 1
                        break
                    else:
                        # 下降失败，增大sigma继续尝试
                        self.sigma *= 2
                        
                except Exception as e:
                    # 任何异常都增大sigma并继续内循环
                    self.sigma *= 2
                
                # 防止sigma过大
                self.sigma = min(self.sigma, 1e12)
            
            # 如果内循环所有尝试都失败，仍然增大sigma并计入迭代
            if not step_accepted:
                # 不更新theta，只增大sigma
                self.sigma *= 2
                iter_count += 1
                print(f"CR: All inner iterations failed at outer iter {iter_count}, sigma increased to {self.sigma:.2e}")
            
            # 防止sigma过大
            self.sigma = min(self.sigma, 1e12)
            
            # 防止无限循环
            if inner_iter_total >= max_iter * self.max_inner_iter:
                print(f"CR: Reached maximum inner iterations ({inner_iter_total})")
                break
        
        return theta



class ARC(BaseOptimizer):
    """自适应正则化立方算法（基于柯西点理论）"""
    
    def __init__(self, eta1=0.1, eta2=0.9, sigma0=1.0, gamma1=2.0, gamma2=0.5, 
                 sigma_min=1e-6, max_solver_iter=200, solver_tol=1e-5):
        super().__init__('ARC')
        self.eta1 = eta1
        self.eta2 = eta2
        self.sigma = sigma0
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.sigma_min = sigma_min
        self.max_solver_iter = max_solver_iter
        self.solver_tol = solver_tol
        self.f_prev = None
    
    def _cauchy_point(self, g, H, M):
        """计算柯西点 - 沿梯度方向的最小值点"""
        if np.linalg.norm(g) == 0 or M == 0:
            return np.zeros_like(g)
        
        g_norm = np.linalg.norm(g)
        g_dir = g / g_norm
        
        # 计算二次项: g_dir^T H g_dir
        H_gg = g_dir.T @ H @ g_dir
        
        # 求解立方正则化问题沿梯度方向的最小值
        # 最小化: g_norm * r + 0.5 * H_gg * r^2 + (M/3) * r^3
        # 导数: g_norm + H_gg * r + M * r^2 = 0
        discriminant = H_gg**2 - 4 * M * g_norm
        
        if discriminant >= 0:
            r1 = (-H_gg + np.sqrt(discriminant)) / (2 * M)
            r2 = (-H_gg - np.sqrt(discriminant)) / (2 * M)
            # 选择正根
            r = max(r1, r2) if r1 > 0 and r2 > 0 else (r1 if r1 > 0 else r2)
        else:
            # 使用近似解
            r = g_norm / (np.abs(H_gg) + M * g_norm)
        
        return -r * g_dir
    
    def _cubic_subsolver(self, x, g, H, M, model):
        """立方正则化子问题求解器"""
        dim = len(g)
        
        # 1. 计算柯西点
        cauchy_step = self._cauchy_point(g, H, M)
        r_min = np.linalg.norm(cauchy_step)
        
        # 2. 计算牛顿步
        try:
            newton_step = -np.linalg.solve(H, g)
            r_max = np.linalg.norm(newton_step)
        except:
            # 如果Hessian奇异，使用梯度方向
            newton_step = -g / (np.linalg.norm(g) + 1e-12)
            r_max = np.linalg.norm(newton_step)
        
        # 检查牛顿步是否直接可用
        if M == 0:
            return x + newton_step, 1
        
        # 定义收敛准则函数
        def convergence_criterion(s, r):
            s_norm = np.linalg.norm(s)
            if s_norm < 1e-12:
                return -1
            return 1/s_norm - 1/r
        
        # 二分法求解最优半径
        identity = np.eye(dim)
        best_step = newton_step
        best_crit = float('inf')
        
        for solver_iter in range(self.max_solver_iter):
            r_try = (r_min + r_max) / 2
            lambda_try = M * r_try
            
            try:
                A = H + lambda_try * jnp.eye(dim)
                step_try = -np.linalg.solve(A, g)
            except:
                # 如果矩阵奇异，增加正则化
                lambda_try *= 2
                A = H + lambda_try * identity + 1e-8 * jnp.eye(dim)
                step_try = -np.linalg.solve(A, g)
            
            crit = convergence_criterion(step_try, r_try)
            
            # 更新最佳步长
            if abs(crit) < abs(best_crit):
                best_step = step_try
                best_crit = crit
            
            if abs(crit) < self.solver_tol:
                break
            
            if crit < 0:
                r_min = r_try
            else:
                r_max = r_try
            
            if r_max - r_min < self.solver_tol:
                break
        
        return x + best_step, solver_iter + 1
    
    def optimize(self, model, dim, initial_theta=None, max_iter=100, tol=1e-8, **kwargs):
        self.reset_history()
        theta = initial_theta if initial_theta is not None else jnp.array(np.random.randn(dim) * 0 + 1.5)
        start_time = time.time()
        
        self.f_prev = model.loss(theta)
        
        for iter_idx in range(max_iter):
            self._record(theta, model, start_time)
            
            # 计算梯度和Hessian
            grad = model.gradient(theta)
            hess = model.hessian(theta)
            grad_norm = np.linalg.norm(grad)
            
            # 检查收敛
            if grad_norm < tol:
                break
            
            # 求解立方正则化子问题
            theta_candidate, solver_iters = self._cubic_subsolver(
                theta, grad, hess, self.sigma, model
            )
            
            # 计算实际下降和预测下降
            f_new = model.loss(theta_candidate)
            
            # 计算模型预测值
            delta_theta = theta_candidate - theta
            model_decrease = (np.dot(grad, delta_theta) + 
                            0.5 * delta_theta.T @ hess @ delta_theta + 
                            self.sigma/3 * np.linalg.norm(delta_theta)**3)
            
            actual_reduction = self.f_prev - f_new
            predicted_reduction = -model_decrease
            
            # 避免除零
            if abs(predicted_reduction) < 1e-12:
                rho = 0.0
            else:
                rho = actual_reduction / abs(predicted_reduction)
            
            # 更新正则化参数和迭代点
            if rho > self.eta1:
                # 成功迭代
                theta = theta_candidate
                self.f_prev = f_new
                
                if rho > self.eta2:
                    # 非常成功迭代，减小正则化
                    self.sigma = max(self.sigma_min, self.sigma * self.gamma2)
            else:
                # 不成功迭代，增大正则化
                self.sigma *= self.gamma1
        
        return theta


class Algorithm1(BaseOptimizer):
    """简化阻尼牛顿法（JAX版本）- 无线搜索版本"""
    def __init__(self, alpha=1/2,beta=1/6,  H0=1, max_inner_iter=200):
        super().__init__('ALM')
        self.alpha = alpha
        self.beta = beta
        self.H0 = H0
        self.max_inner_iter = max_inner_iter
        self.history_theta = []
        self.history_grad = []
    
    def optimize(self, model, dim, initial_theta=None, max_iter=100, tol=1e-8, **kwargs):
        self.reset_history()
        theta = initial_theta if initial_theta is not None else jnp.array(np.random.randn(dim) * 0)
        start_time = time.time()
        H_t = self.H0
        
        for t in range(max_iter):
            self._record(theta, model, start_time)
            grad = model.gradient(theta)
            grad_norm = jnp.linalg.norm(grad)
            if grad_norm < tol:
                break
                
            j_t = 0
            success = False
            hess = model.hessian(theta)
            for _ in range(self.max_inner_iter):
                lambda_jt = (2**(j_t)) * H_t
                try:
                    # 计算阻尼牛顿方向
                    A = hess + lambda_jt * jnp.eye(dim)
                    L = jax.scipy.linalg.cho_factor(A, lower=True)
                    d_plus = jax.scipy.linalg.cho_solve(L, -grad)
                    
                    #不进行线搜索
                    theta_candidate = theta + d_plus
                    candidate_loss = model.loss(theta_candidate)
                    current_loss = model.loss(theta)
                    expected_reduction = self.alpha * jnp.dot(grad, d_plus)-lambda_jt*self.beta*jnp.dot(d_plus, d_plus)
                    condition1= candidate_loss <= current_loss + expected_reduction
                    # 检查条件
                    if condition1:
                        # 接受这一步
                        theta = theta_candidate
                        self.history_theta.append(theta.copy())
                        self.history_grad.append(grad.copy())
                        # 调整正则化参数
                        H_t = max((2**(j_t) * H_t) / 2, 1e-8)
                        success = True
                        break
                    else:
                        # Armijo条件不满足，增加阻尼系数
                        j_t += 1
                        continue
                        
                except Exception as e:
                    # 矩阵分解失败，增加阻尼系数
                    j_t += 1
            
            if not success:
                # 内循环所有尝试都失败，接受当前候选解（如果有）或保持原值
                if 'theta_candidate' in locals():
                    theta = theta_candidate
                self.history_theta.append(theta.copy())
                self.history_grad.append(grad.copy())
                
        return theta

class SuperUniversalNewton(BaseOptimizer):
    """超级通用牛顿法（JAX版本）"""
    def __init__(self, H_0=1.0, alpha=0.75, adaptive_search=True, H_min=1e-5):
        super().__init__('SUN')
        self.H_0 = H_0
        self.alpha = alpha
        self.adaptive_search = adaptive_search
        self.H_min = H_min
    
    def optimize(self, model, dim, initial_theta=None, max_iter=100, tol=1e-8, **kwargs):
        self.reset_history()
        theta = initial_theta if initial_theta is not None else jnp.array(np.random.randn(dim) * 0 + 1.5)
        start_time = time.time()
        
        f_k = model.loss(theta)
        g_k = model.gradient(theta)
        g_k_norm = jnp.linalg.norm(g_k)
        H_k = self.H_0
        
        for k in range(max_iter):
            self._record(theta, model, start_time)
            
            if g_k_norm < tol:
                break
            
            Hess_k = model.hessian(theta)

            adaptive_search_max_iter = 40
            for i in range(adaptive_search_max_iter + 1):
                if i == adaptive_search_max_iter:
                    break

                lambda_k = H_k * (g_k_norm ** self.alpha)
                
                try:
                    A = Hess_k + lambda_k * jnp.eye(dim)
                    L = jax.scipy.linalg.cho_factor(A, lower=True)
                    delta_theta = jax.scipy.linalg.cho_solve(L, -g_k)
                except:
                    H_k *= 4
                    continue
                
                theta_new = theta + delta_theta
                f_new = model.loss(theta_new)
                g_new = model.gradient(theta_new)
                g_new_norm = jnp.linalg.norm(g_new)

                if not self.adaptive_search:
                    break

                lhs = jnp.dot(g_new, -delta_theta)
                rhs = (g_new_norm ** 2) / (4* lambda_k)
                
                if lhs >= rhs:
                    H_k = jnp.maximum(H_k * 0.25, self.H_min)
                    break
                
                H_k *= 4

            theta = theta_new
            f_k = f_new
            g_k = g_new
            g_k_norm = g_new_norm
            
        return theta
        
        
class CubicMM(BaseOptimizer):
    """固定L的Cubic MM算法 - 无自适应调整"""
    
    def __init__(self, L_fixed=10.0):
        """
        Args:
            L_fixed: 固定的Lipschitz常数
        """
        super().__init__('CMM')
        self.L_fixed = L_fixed
        self.history_L = []
        self.history_success = []
    
    def optimize(self, model, dim, initial_theta=None, max_iter=100, tol=1e-8, **kwargs):
        self.reset_history()
        self.history_L = []
        self.history_success = []
        
        theta = initial_theta if initial_theta is not None else jnp.array(np.random.randn(dim))
        start_time = time.time()
        L_k = self.L_fixed  # 固定使用初始L值
        
        for k in range(max_iter):
            self._record(theta, model, start_time)
            
            # 计算当前梯度和Hessian
            grad = model.gradient(theta)
            grad_norm = jnp.linalg.norm(grad)
            hess = model.hessian(theta)
            
            if grad_norm < tol:
                break
            
            # 计算最小特征值用于Hessian修正
            try:
                lambda_min = jax.scipy.linalg.eigh(hess, subset_by_index=[0, 0])[0][0]
                if lambda_min <= 0:
                    lambda_val = lambda_min - 1e-3
                else:
                    lambda_val = 0.0
            except:
                lambda_val = 0.0
            
            inner_success = False
            
            # 计算信任区域参数（使用固定L）
            c = jnp.sqrt(3.0 / L_k) 
            g = jnp.sqrt(grad_norm)
            r = c * g
            d = jnp.maximum(L_k * c / 3 , 1.0 / c)
            
            # 构建修正Hessian
            H = hess + (-lambda_val + d * g) * jnp.eye(dim)
            
            # 求解线性系统
            try:
                L_chol = jax.scipy.linalg.cho_factor(H, lower=True)
                v = jax.scipy.linalg.cho_solve(L_chol, -grad)
                
                # 检查步长
                step_norm = jnp.linalg.norm(v)
                if step_norm <= r + 1e-8:
                    # 步长在信任区域内，接受更新
                    theta_candidate = theta + v
                    f_current = model.loss(theta)
                    f_candidate = model.loss(theta_candidate)
                    
                    # 检查函数值下降
                    if f_candidate < f_current - 1e-12:
                        theta = theta_candidate
                        inner_success = True
                
            except:
                # Cholesky失败，保持theta不变
                L_k=L_k*2
                pass
            
            # 记录历史（L值始终固定）
            self.history_L.append(float(L_k))
            self.history_success.append(1 if inner_success else 0)
        
        return theta
    
    def get_detailed_history(self):
        """获取详细的优化历史"""
        return {
            'loss': self.history.get('loss', []),
            'grad_norm': self.history.get('grad_norm', []),
            'time': self.history.get('time', []),
            'L_values': self.history_L,
            'success_flags': self.history_success
        }
