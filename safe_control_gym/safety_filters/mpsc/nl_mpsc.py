'''NL Model Predictive Safety Certification (NL MPSC).

The core idea is that any learning controller input can be either certificated as safe or, if not safe, corrected
using an MPC controller based on Robust NL MPC.

Based on
    * K.P. Wabsersich and M.N. Zeilinger 'Linear model predictive safety certification for learning-based control' 2019
      https://arxiv.org/pdf/1803.08552.pdf
    * J. Köhler, R. Soloperto, M. A. Müller, and F. Allgöwer, “A computationally efficient robust model predictive
      control framework for uncertain nonlinear systems -- extended version,” IEEE Trans. Automat. Contr., vol. 66,
      no. 2, pp. 794 801, Feb. 2021, doi: 10.1109/TAC.2020.2982585. http://arxiv.org/abs/1910.12081
'''

import pickle

import numpy as np
import casadi as cs
import cvxpy as cp
from scipy.linalg import sqrtm, solve_discrete_are

from safe_control_gym.safety_filters.mpsc.mpsc import MPSC
from safe_control_gym.controllers.mpc.mpc_utils import rk_discrete
from safe_control_gym.safety_filters.mpsc.mpsc_utils import Cost_Function
from safe_control_gym.safety_filters.cbf.cbf_utils import cartesian_product
from safe_control_gym.envs.benchmark_env import Task, Environment


class NL_MPSC(MPSC):
    '''Model Predictive Safety Certification Class.'''

    def __init__(self,
                 env_func,
                 horizon: int = 10,
                 q_lin: list = None,
                 r_lin: list = None,
                 integration_algo: str = 'rk4',
                 warmstart: bool = True,
                 additional_constraints: list = None,
                 use_terminal_set: bool = True,
                 n_samples: int = 600,
                 cost_function: Cost_Function = Cost_Function.ONE_STEP_COST,
                 mpsc_cost_horizon: int = 5,
                 decay_factor: float = 0.85,
                 **kwargs
                 ):
        '''Initialize the MPSC.

        Args:
            env_func (partial BenchmarkEnv): Environment for the task.
            horizon (int): The MPC horizon.
            integration_algo (str): The algorithm used for integrating the dynamics,
                either 'rk4', 'rk', or 'cvodes'.
            warmstart (bool): If the previous MPC solution should be used to warmstart the next mpc step.
            additional_constraints (list): List of additional constraints to consider.
            use_terminal_set (bool): Whether to use a terminal set constraint or not.
            n_samples (int): The number of state/action pairs to test when determining w_func.
            cost_function (Cost_Function): A string (from Cost_Function) representing the cost function to be used.
            mpsc_cost_horizon (int): How many steps forward to check for constraint violations.
            decay_factor (float): How much to discount future costs.
        '''

        super().__init__(env_func, horizon, q_lin, r_lin, integration_algo, warmstart, additional_constraints, use_terminal_set, cost_function, mpsc_cost_horizon, decay_factor, **kwargs)

        self.n_samples = n_samples

        self.n = self.model.nx
        self.m = self.model.nu
        self.q = self.model.nx

        self.state_constraint = self.constraints.state_constraints[0]
        self.input_constraint = self.constraints.input_constraints[0]

        [self.X_mid, L_x, l_x] = self.box2polytopic(self.state_constraint)
        [self.U_mid, L_u, l_u] = self.box2polytopic(self.input_constraint)

        # number of constraints
        p_x = l_x.shape[0]
        p_u = l_u.shape[0]
        self.p = p_x + p_u

        self.L_x = np.vstack((L_x, np.zeros((p_u, self.n))))
        self.L_u = np.vstack((np.zeros((p_x, self.m)), L_u))
        self.l = np.concatenate([l_x, l_u])

    def set_dynamics(self):
        '''Compute the discrete dynamics.'''

        if self.integration_algo == 'rk4':
            dynamics_func = rk_discrete(self.model.fc_func,
                                        self.model.nx,
                                        self.model.nu,
                                        self.dt)
        else:
            dynamics_func = cs.integrator('fd', self.integration_algo,
                                          {'x': self.model.x_sym,
                                           'p': self.model.u_sym,
                                           'ode': self.model.x_dot}, {'tf': self.dt}
                                          )

        self.dynamics_func = dynamics_func

    def learn(self,
              env=None,
              **kwargs
              ):
        '''Compute values used by the MPC.

        Args:
            env (BenchmarkEnv): If a different environment is to be used for learning, can supply it here.
        '''

        if env is None:
            env = self.training_env

        self.tolerance = 1e-4

        if self.env.NAME == Environment.CARTPOLE:
            self.x_r = np.array([self.X_EQ[0], 0, 0, 0])
        elif self.env.NAME == Environment.QUADROTOR and self.env.QUAD_TYPE == 2:
            self.x_r = np.array([self.X_EQ[0], 0, self.X_EQ[2], 0, 0, 0])
        elif self.env.NAME == Environment.QUADROTOR and self.env.QUAD_TYPE == 3:
            self.x_r = np.array([self.X_EQ[0], 0, self.X_EQ[2], 0, self.X_EQ[4], 0, 0, 0, 0, 0, 0, 0])
        self.u_r = self.U_EQ

        x_sym = self.model.x_sym
        u_sym = self.model.u_sym
        w_sym = cs.MX.sym('delta_w', self.q, 1)

        self.get_error_function(env=env)
        self.E = np.diag(self.max_w_per_dim) / self.dt

        if self.env.NAME == Environment.QUADROTOR and self.env.QUAD_TYPE == 2:
            self.E *= 2

        self.f = cs.Function('f', [x_sym, u_sym, w_sym], [self.model.fc_func(x_sym + self.X_mid, u_sym + self.U_mid) + self.E @ w_sym], ['x', 'u', 'w'], ['f'])
        phi_1 = cs.Function('phi_1', [x_sym, u_sym, w_sym], [self.f(x_sym, u_sym, w_sym)], ['x', 'u', 'w'], ['phi_1'])
        phi_2 = cs.Function('phi_2', [x_sym, u_sym, w_sym], [self.f(x_sym + 0.5 * self.dt * phi_1(x_sym, u_sym, w_sym), u_sym, w_sym)], ['x', 'u', 'w'], ['phi_2'])
        phi_3 = cs.Function('phi_3', [x_sym, u_sym, w_sym], [self.f(x_sym + 0.5 * self.dt * phi_2(x_sym, u_sym, w_sym), u_sym, w_sym)], ['x', 'u', 'w'], ['phi_3'])
        phi_4 = cs.Function('phi_4', [x_sym, u_sym, w_sym], [self.f(x_sym + self.dt * phi_3(x_sym, u_sym, w_sym), u_sym, w_sym)], ['x', 'u', 'w'], ['phi_4'])
        rungeKutta = x_sym + self.dt / 6 * (phi_1(x_sym, u_sym, w_sym) + 2 * phi_2(x_sym, u_sym, w_sym) + 2 * phi_3(x_sym, u_sym, w_sym) + phi_4(x_sym, u_sym, w_sym))
        self.disc_f = cs.Function('disc_f', [x_sym, u_sym, w_sym], [rungeKutta + self.X_mid], ['x', 'u', 'w'], ['disc_f'])

        self.Ac = cs.Function('Ac', [x_sym, u_sym, w_sym], [cs.jacobian(self.f(x_sym, u_sym, w_sym), x_sym)], ['x', 'u', 'w'], ['Ac'])
        self.Bc = cs.Function('Bc', [x_sym, u_sym, w_sym], [cs.jacobian(self.f(x_sym, u_sym, w_sym), u_sym)], ['x', 'u', 'w'], ['Bc'])

        self.Ad = cs.Function('Ad', [x_sym, u_sym, w_sym], [cs.jacobian(self.disc_f(x_sym, u_sym, w_sym), x_sym)], ['x', 'u', 'w'], ['Ad'])
        self.Bd = cs.Function('Bd', [x_sym, u_sym, w_sym], [cs.jacobian(self.disc_f(x_sym, u_sym, w_sym), u_sym)], ['x', 'u', 'w'], ['Bd'])

        self.synthesize_lyapunov()
        self.get_terminal_ingredients()

        self.L_x_sym = cs.MX(self.L_x)
        self.L_u_sym = cs.MX(self.L_u)
        self.l_sym = cs.MX(self.l)
        self.setup_optimizer()

    def get_error_function(self, env):
        '''Computes the maximum disturbance found in the training environment.

        Args:
            env (BenchmarkEnv): If a different environment is to be used for learning, can supply it here.
        '''

        if env is None:
            env = self.training_env

        # Create set of error residuals.
        w = np.zeros((self.n_samples, self.n))

        # Use uniform sampling of control inputs and states.
        for i in range(self.n_samples):
            init_state, _ = env.reset()
            if self.env.NAME == Environment.QUADROTOR:
                u = np.random.rand(self.model.nu) / 20 - 1 / 40 + self.U_EQ
            else:
                u = env.action_space.sample()  # Will yield a random action within action space.
            x_next_obs, _, _, _ = env.step(u)
            x_next_estimate = np.squeeze(self.dynamics_func(x0=init_state, p=u)['xf'].toarray())
            w[i, :] = x_next_obs - x_next_estimate

        normed_w = np.linalg.norm(w, axis=1)
        print('MAX ERROR:', np.max(normed_w))
        print('MEAN ERROR:', np.mean(normed_w))
        print('MAX ERROR PER DIM:', np.max(w, axis=0))
        print('TOTAL ERRORS BY CHANNEL:', np.sum(np.abs(w), axis=0))
        self.error_distrib = np.sum(np.abs(w), axis=0) / np.linalg.norm(np.sum(np.abs(w), axis=0))
        self.max_w_per_dim = np.max(w, axis=0)
        self.max_w = np.max(normed_w)
        self.w_func = lambda x, u, s: self.max_w

    def synthesize_lyapunov(self):
        '''Synthesize the appropriate constants related to the lyapunov function of the system.'''
        # Incremental Lyapunov function: Find upper bound for S-procedure variable lambda
        lamb_lb = None
        lamb_ub = None

        lamb = 0.008  # lambda lower bound
        self.rho_c = 0.192  # tuning parameter determines how fast the lyapunov function contracts

        if self.env.NAME == Environment.CARTPOLE or (self.env.NAME == Environment.QUADROTOR and self.env.QUAD_TYPE == 2):
            self.Theta = [self.state_constraint.lower_bounds[-2], 0, self.state_constraint.upper_bounds[-2]]
        else:
            self.Theta = [self.state_constraint.lower_bounds[6], 0, self.state_constraint.upper_bounds[6]]

        while lamb < 100:
            lamb = lamb * 2
            [X, Y, cost, constraints] = self.setup_tube_optimization(lamb)
            prob = cp.Problem(cp.Minimize(cost), constraints)
            try:
                print(f'Attempting with lambda={lamb}.')
                cost = prob.solve(solver=cp.MOSEK, verbose=False)
                if prob.status == 'optimal' and cost != float('inf'):
                    print(f'Succeeded with cost={cost}.')
                    if lamb_lb is None:
                        lamb_lb = lamb
                    lamb_ub = lamb
                else:
                    raise Exception('Not optimal or cost is infinite.')
            except Exception as e:
                print('Error in optimization:', e)
                if lamb_lb is not None:
                    break

        # Incremental Lyapunov function: Determine optimal lambda
        lamb_lb = lamb_lb / 2
        lamb_ub = lamb_ub * 2

        num_candidates = 50

        lambda_candidates = np.logspace(np.log(lamb_lb) / np.log(10), np.log(lamb_ub) / np.log(10), num_candidates)
        cost_values = []

        for i in range(num_candidates):
            lambda_candidate = lambda_candidates[i]
            [X, Y, cost, constraints] = self.setup_tube_optimization(lambda_candidate)
            prob = cp.Problem(cp.Minimize(cost), constraints)
            try:
                cost = prob.solve(solver=cp.MOSEK, verbose=False)
                if prob.status != 'optimal' or cost == float('inf'):
                    raise cp.SolverError
            except Exception as e:
                print('Error in optimization:', e)
                cost = float('inf')
            cost_values += [cost]

        best_index = cost_values.index(min(cost_values))
        best_lamb = lambda_candidates[best_index]
        [X, Y, cost, constraints] = self.setup_tube_optimization(best_lamb)
        prob = cp.Problem(cp.Minimize(cost), constraints)
        cost = prob.solve(solver=cp.MOSEK, verbose=False)
        if prob.status != 'optimal' or cost == float('inf'):
            raise cp.SolverError

        # Resulting continuous-time parameters
        self.X = X.value
        self.P = np.linalg.pinv(self.X)
        self.K = Y.value @ self.P

        self.c_js = np.zeros(self.p)

        for j in range(self.p):
            self.c_js[j] = np.linalg.norm((self.L_x[j, :] + self.L_u[j, :] @ self.K) @ sqrtm(self.X))

        c_max = max(self.c_js)
        w_bar_c = np.sqrt(np.max(np.linalg.eig(self.E.T @ self.P @ self.E)[0]))

        # Get Discrete-time system values
        self.rho = np.exp(-self.rho_c * self.dt)
        self.w_bar = w_bar_c * (1 - self.rho) / self.rho_c  # even using rho_c from the paper yields different w_bar
        horizon_multiplier = (1 - self.rho**self.horizon) / (1 - self.rho)
        self.s_bar_f = horizon_multiplier * self.w_bar
        assert self.w_bar > self.max_w, f'[ERROR] w_bar ({self.w_bar}) is too small compared to max_w ({self.max_w}).'
        assert self.s_bar_f > self.max_w * horizon_multiplier + self.tolerance, f'[ERROR] s_bar_f ({self.s_bar_f}) is too small with respect to max_w ({self.max_w}).'
        assert self.max_w * horizon_multiplier < 1.0, '[ERROR] max_w is too large and will overwhelm terminal set.'
        self.s_bar_f = self.max_w * horizon_multiplier + self.tolerance
        self.gamma = 1 / c_max - self.s_bar_f

        self.delta_loc = (horizon_multiplier * self.w_bar)**2

        print(f'rho: {self.rho}')
        print(f'w_bar: {self.w_bar}')
        print(f'Original s_bar_f: {self.w_bar * horizon_multiplier}')
        print(f's_bar_f: {self.s_bar_f}')
        print(f'gamma: {self.gamma}')

        self.check_decay_rate()
        self.check_lyapunov_func()

    def get_terminal_ingredients(self):
        '''Calculate the terminal ingredients of the MPC optimization.'''
        # Solve Lyapunov SDP using linearized discrete-time dynamics based on RK4 for terminal ingredients
        w_none = np.zeros((self.q, 1))
        A_lin = self.Ad(self.x_r - self.X_mid, self.u_r - self.U_mid, w_none).toarray()
        B_lin = self.Bd(self.x_r - self.X_mid, self.u_r - self.U_mid, w_none).toarray()

        self.P_f = solve_discrete_are(A_lin, B_lin, self.Q, self.R)
        btp = np.dot(B_lin.T, self.P_f)
        self.K_f = -np.dot(np.linalg.inv(self.R + np.dot(btp, B_lin)), np.dot(btp, A_lin))
        self.check_terminal_ingredients()
        self.check_terminal_constraints()

    def box2polytopic(self, constraint):
        '''Convert constraints into an explicit polytopic form. This assumes that constraints contain the origin.

        Args:
            constraint (Constraint): The constraint to be converted.

        Returns:
            L (ndarray): The polytopic matrix.
            l (ndarray): Whether the constraint is active.
        '''

        L = []
        l = []

        Z_mid = (constraint.upper_bounds + constraint.lower_bounds) / 2.0
        Z_limits = np.array([[constraint.upper_bounds[i] - Z_mid[i], constraint.lower_bounds[i] - Z_mid[i]] for i in range(constraint.upper_bounds.shape[0])])

        dim = Z_limits.shape[0]
        eye_dim = np.eye(dim)

        for constraint_id in range(0, dim):
            if Z_limits[constraint_id, 0] != -float('inf'):
                if Z_limits[constraint_id, 0] == 0:
                    l += [0]
                    L += [-eye_dim[constraint_id, :]]
                else:
                    l += [1]
                    factor = 1 / Z_limits[constraint_id, 0]
                    L += [factor * eye_dim[constraint_id, :]]

            if Z_limits[constraint_id, 1] != float('inf'):
                if Z_limits[constraint_id, 1] == 0:
                    l += [0]
                    L += [eye_dim[constraint_id, :]]
                else:
                    l += [1]
                    factor = 1 / Z_limits[constraint_id, 1]
                    L += [factor * eye_dim[constraint_id, :]]

        return Z_mid, np.array(L), np.array(l)

    def setup_tube_optimization(self, lamb):
        '''Sets up the optimization to find the lyapunov function.

        Args:
            lamb (float): The S-procedure constant.

        Returns:
            X (cp.Variable): The X variable in the optimization.
            Y (cp.Variable): The Y variable in the optimization.
            Cost (cp.Expression): The cost function expression.
            Constraints (list): The list of cvxpy expressions representing the constraints.
        '''

        X = cp.Variable((self.n, self.n), PSD=True, name='X', complex=False)
        Y = cp.Variable((self.m, self.n), name='Y', complex=False)

        Cost = -cp.log_det(X)

        Constraints = []

        x_test = np.zeros((self.n, 1))
        u_test = self.U_EQ
        w_test = np.zeros((self.q, 1))

        for angle in self.Theta:
            if self.env.NAME == Environment.CARTPOLE or (self.env.NAME == Environment.QUADROTOR and self.env.QUAD_TYPE == 2):
                x_test[-2] = angle
            else:
                x_test[-4] = angle
                x_test[-5] = angle
                x_test[-6] = angle
            A_theta = self.Ac(x_test, u_test - self.U_mid, w_test).toarray()
            B_theta = self.Bc(x_test, u_test - self.U_mid, w_test).toarray()

            AXBY = A_theta @ X + B_theta @ Y

            constraint_1 = AXBY + AXBY.T + 2 * self.rho_c * X
            constraint_2 = cp.bmat([[AXBY + AXBY.T + lamb * X, self.E], [self.E.T, -lamb * np.eye(self.q)]])

            Constraints += [constraint_1 << 0]
            Constraints += [constraint_2 << 0]

        for j in range(0, self.p):
            LXLY = self.L_x[j:j + 1, :] @ X + self.L_u[j:j + 1, :] @ Y

            constraint_3 = cp.bmat([[np.array([[1]]), LXLY], [LXLY.T, X]])
            Constraints += [constraint_3 >> 0]

        return X, Y, Cost, Constraints

    def randsphere(self, num, dim, r):
        '''This function returns an num by dim array in which
           each of the num rows has the dim Cartesian coordinates
           of a random point uniformly-distributed over the
           interior of an dim-dimensional hypersphere with
           radius r and center at the origin.

        Args:
            num (int): The number of vectors.
            dim (int): The dimension of the hypersphere.
            r (float): The radius of the hypersphere.

        Returns:
            vectors (ndarray): The resulting random points inside the hypersphere.
        '''

        vectors = []

        while len(vectors) < num:
            u = np.random.normal(0, 1, dim)  # an array of d normally distributed random variables
            norm = np.sum(u**2)**(0.5)
            radius = r * np.random.rand()**(1.0 / dim)
            vec = radius * u / norm
            vectors.append(vec)

        return np.vstack(vectors)

    def check_decay_rate(self):
        '''Check the decay rate.'''

        x_test = np.zeros((self.n, 1))
        u_test = self.U_EQ
        w_test = np.zeros((self.q, 1))

        X_sqrt = sqrtm(self.X)
        P_sqrt = sqrtm(self.P)
        for angle in self.Theta:
            if self.env.NAME == Environment.CARTPOLE or (self.env.NAME == Environment.QUADROTOR and self.env.QUAD_TYPE == 2):
                x_test[-2] = angle
            else:
                x_test[-4] = angle
                x_test[-5] = angle
                x_test[-6] = angle
            A_theta = self.Ac(x_test, u_test - self.U_mid, w_test).toarray()
            B_theta = self.Bc(x_test, u_test - self.U_mid, w_test).toarray()
            left_side = max(np.linalg.eig(X_sqrt @ (A_theta + B_theta @ self.K).T @ P_sqrt + P_sqrt @ (A_theta + B_theta @ self.K) @ X_sqrt)[0]) + 2 * self.rho_c
            assert left_side <= self.tolerance, f'[ERROR] The solution {left_side} is not within the tolerance {self.tolerance}'

    def check_lyapunov_func(self):
        '''Check the incremental Lyapunov function.'''

        # select the number of random vectors to check
        num_random_vectors = 10000

        # Sample random points inside the set V_delta(x, z) <= delta_loc
        delta_x = self.randsphere(num_random_vectors, self.n, self.delta_loc).T
        dx_transform = np.linalg.inv(sqrtm(self.P)) @ delta_x
        dx_transform = self.x_r[:, None] + dx_transform  # transform point from error to actual state

        # sample random disturbance bounded by max_w
        w_dist = self.randsphere(num_random_vectors, self.q, self.max_w).T

        # set arbitrary v that satisfies the constraints for testing
        v = np.array(self.constraints.input_constraints[0].upper_bounds) / 10

        # initialize counters
        num_valid = 0
        inside_set = 0
        is_invariant = 0

        for i in range(num_random_vectors):
            # get random state
            x_i = dx_transform[:, i]

            # set up control inputs (u_r is required to get f_kappa(0, 0) = 0)
            u_x = self.K @ x_i + v + self.u_r
            u_z = self.K @ self.x_r + v + self.u_r

            # get dynamics
            w_none = np.zeros((self.q, 1))
            x_dot = np.squeeze(self.f(x_i - self.X_mid, u_x - self.U_mid, w_none).toarray())
            z_dot = np.squeeze(self.f(self.x_r - self.X_mid, u_z - self.U_mid, w_none).toarray())

            # evaluate Lyapunov function and its time derivative
            V_d = (x_i - self.x_r).T @ self.P @ (x_i - self.x_r)
            dVdt = (x_i - self.x_r).T @ self.P @ (x_dot - z_dot)

            # Check incremental Lypaunov function condition
            if dVdt <= -self.rho_c * V_d:
                num_valid += 1

            # check if states are inside V_d(x_i, z) <= delta_loc
            if V_d <= self.delta_loc:
                inside_set += 1

            # get next state
            x_plus = np.squeeze(self.disc_f(x_i - self.X_mid, u_x - self.U_mid, w_dist[:, i]).toarray())
            V_d_plus = (x_plus - self.x_r).T @ self.P @ (x_plus - self.x_r)

            # check robust control invariance
            if V_d_plus <= self.delta_loc:
                is_invariant += 1

        print('NUM_VALID:', num_valid / num_random_vectors)
        print('INSIDE SET:', inside_set / num_random_vectors)
        print('IS INVARIANT:', is_invariant / num_random_vectors)

    def check_terminal_ingredients(self):
        '''Check the terminal ingredients.'''

        w_none = np.zeros((self.q, 1))
        num_random_vectors = 10000

        # Sample points from gamma^2 * unit sphere
        delta_x = self.randsphere(num_random_vectors, self.n, self.gamma**2).T

        # Transform sampled points into ellipsoid to span the candidate terminal
        # set and shift around reference point x_r
        dx_transform = np.linalg.inv(sqrtm(self.P_f)) @ delta_x
        dx_transform = self.x_r[:, None] + dx_transform

        # sample random disturbance bounded by max_w
        w_dist = self.randsphere(num_random_vectors, self.q, self.max_w).T

        # initialize counter
        num_valid = 0
        inside_set = 0

        for i in range(num_random_vectors):
            # get sampled vector
            x_i = dx_transform[:, i]

            # get terminal control input
            u = self.K_f @ (x_i - self.x_r) + self.u_r

            # simulate system using control input
            x_plus = np.squeeze(self.disc_f(x_i - self.X_mid, u - self.U_mid, w_none).toarray())

            # disturbed x_plus
            x_plus_noisy = np.squeeze(self.disc_f(x_i - self.X_mid, u - self.U_mid, w_dist[:, i]).toarray())

            # evaluate stage cost and terminal costs
            stage = (x_i - self.x_r).T @ self.Q @ (x_i - self.x_r)
            V_f = (x_i - self.x_r).T @ self.P_f @ (x_i - self.x_r)
            V_f_plus = (x_plus - self.x_r).T @ self.P_f @ (x_plus - self.x_r)

            # check Lyapunov condition for terminal cost
            if V_f_plus <= V_f - stage:
                num_valid += 1

            # check if noisy state is still in terminal set
            V_f_plus_noisy = (x_plus_noisy - self.x_r).T @ self.P_f @ (x_plus_noisy - self.x_r)
            if V_f_plus_noisy <= self.gamma**2:
                inside_set += 1

        print('NUM_VALID:', num_valid / num_random_vectors)
        print('INSIDE SET:', inside_set / num_random_vectors)

    def check_terminal_constraints(self,
                                   num_points: int = 40,
                                   ):
        '''
        Check if the provided terminal set is only contains valid states using a gridded approach.

        Args:
            num_points (int): The number of points in each dimension to check.

        Returns:
            valid_cbf (bool): Whether the provided CBF candidate is valid.
            infeasible_states (list): List of all states for which the QP is infeasible.
        '''

        # Determine if terminal set inside state constraints
        terminal_max = np.sqrt(np.diag(np.linalg.inv(self.P_f / self.gamma**2)))
        terminal_min = -np.sqrt(np.diag(np.linalg.inv(self.P_f / self.gamma**2)))

        max_bounds = np.zeros((self.n))
        min_bounds = np.zeros((self.n))
        for i in range(self.n):
            tighten_by_max = self.c_js[i * 2] * self.s_bar_f
            tighten_by_min = self.c_js[i * 2 + 1] * self.s_bar_f
            max_bounds[i] = 1.0 / self.L_x[i * 2, i] * (self.l[i * 2] - tighten_by_max)
            min_bounds[i] = 1.0 / self.L_x[i * 2 + 1, i] * (self.l[i * 2 + 1] - tighten_by_min)

        if np.any(terminal_max > max_bounds) or np.any(terminal_min < min_bounds):
            raise ValueError('Terminal set is not constrained within the constraint set.')

        # Determine if the maximum input is within input constraints
        x = cp.Variable((self.n, 1))
        C = np.linalg.cholesky(self.P_f).T
        cost = cp.Maximize(self.K_f[0, :] @ x)
        constraint = [cp.norm(C @ x) <= self.gamma]
        prob = cp.Problem(cost, constraint)
        max_input = prob.solve(solver=cp.MOSEK)

        max_bounds = np.zeros((self.m))
        min_bounds = np.zeros((self.m))
        for i in range(self.m):
            tighten_by_max = self.c_js[self.n * 2 + i * 2] * self.s_bar_f
            tighten_by_min = self.c_js[self.n * 2 + i * 2 + 1] * self.s_bar_f
            max_bounds[i] = 1.0 / self.L_u[self.n * 2 + i * 2, i] * (self.l[self.n * 2 + i * 2] - tighten_by_max)
            min_bounds[i] = 1.0 / self.L_u[self.n * 2 + i * 2 + 1, i] * (self.l[self.n * 2 + i * 2 + 1] - tighten_by_min)

        if np.any(max_input + self.u_r > max_bounds + self.U_mid) or np.any(-max_input + self.u_r < min_bounds + self.U_mid):
            raise ValueError(f'Terminal controller causes inputs (max_input: {-max_input+self.u_r[0]}/{max_input+self.u_r[0]}) outside of input constraints (constraints: {min_bounds[0] + self.U_mid[0]}/{max_bounds[0] + self.U_mid[0]}).')

        # Make sure that every vertex is checked
        num_points = max(2 * self.n, num_points + num_points % (2 * self.n))
        num_points_per_dim = num_points // self.n

        # Create the lists of states to check
        states_to_sample = [np.linspace(self.X_mid[i], terminal_max[i] + self.X_mid[i], num_points_per_dim) for i in range(self.n)]
        states_to_check = cartesian_product(*states_to_sample)

        num_states_inside_set = 0
        failed_checks = 0
        failed_29a = 0
        failed_29b = 0
        failed_29d = 0

        for state in states_to_check:
            terminal_cost = (state - self.X_mid).T @ self.P_f @ (state - self.X_mid)
            in_terminal_set = terminal_cost < self.gamma**2

            if in_terminal_set:
                num_states_inside_set += 1
                failed = False

                # Testing condition 29a
                stable_input = self.K_f @ (state - self.x_r) + self.u_r
                next_state = np.squeeze(self.disc_f(state - self.X_mid, stable_input - self.U_mid, np.zeros((self.q, 1))).toarray())
                stage_cost = (state.T - self.X_mid) @ self.Q @ (state - self.X_mid)
                next_terminal_cost = (next_state - self.X_mid).T @ self.P_f @ (next_state - self.X_mid)

                if terminal_cost - stage_cost != 0 and next_terminal_cost / (terminal_cost - stage_cost) > 1.01:
                    failed_29a += 1
                    failed = True

                # Testing condition 29b
                num_disturbances = 100
                disturbances = self.randsphere(num_disturbances, self.n, self.max_w).T
                for w in range(num_disturbances):
                    disturbed_state = next_state + disturbances[:, w]
                    terminal_cost = (disturbed_state - self.X_mid).T @ self.P_f @ (disturbed_state - self.X_mid)
                    in_terminal_set = terminal_cost < self.gamma**2

                    if not in_terminal_set:
                        failed_29b += 1
                        failed = True
                        break

                # Testing condition 29d
                for j in range(self.p):
                    constraint_satisfaction = self.L_x[j, :] @ (state - self.X_mid) + self.L_u[j, :] @ (stable_input - self.U_mid) - self.l[j] + self.c_js[j] * self.s_bar_f <= 0
                    if not constraint_satisfaction:
                        failed_29d += 1
                        failed = True
                        break

                if failed:
                    failed_checks += 1

        print(f'Number of states checked: {len(states_to_check)}')
        print(f'Number of states inside terminal set: {num_states_inside_set}')
        print(f'Number of checks failed: {failed_checks}')
        print(f'Number of checks failed due to 29a: {failed_29a}')
        print(f'Number of checks failed due to 29b: {failed_29b}')
        print(f'Number of checks failed due to 29d: {failed_29d}')

    def load(self,
             path,
             ):
        '''Load values used by the MPSC.

        Args:
            path (str): Path to the required file.
        '''

        with open(path, 'rb') as f:
            parameters = pickle.load(f)

        print(parameters)

        self.rho_c = parameters['rho_c']
        self.Theta = parameters['Theta']
        self.X = parameters['X']
        self.K = parameters['K']
        self.P = parameters['P']
        self.delta_loc = parameters['delta_loc']
        self.rho = parameters['rho']
        self.s_bar_f = parameters['s_bar_f']
        self.w_bar = parameters['w_bar']
        self.max_w = parameters['max_w']
        self.w_func = lambda x, u, s: self.max_w
        self.c_js = parameters['c_js']
        self.gamma = parameters['gamma']
        self.P_f = parameters['P_f']
        self.K_f = parameters['K_f']

        self.L_x_sym = cs.MX(self.L_x)
        self.L_u_sym = cs.MX(self.L_u)
        self.l_sym = cs.MX(self.l)

        self.setup_optimizer()

    def save(self, path):
        '''Save values used by the MPSC.

        Args:
            path (str): Name of the file to be created.
        '''

        parameters = {}
        parameters['rho_c'] = self.rho_c
        parameters['Theta'] = self.Theta
        parameters['X'] = self.X
        parameters['K'] = self.K
        parameters['P'] = self.P
        parameters['delta_loc'] = self.delta_loc
        parameters['rho'] = self.rho
        parameters['s_bar_f'] = self.s_bar_f
        parameters['w_bar'] = self.w_bar
        parameters['max_w'] = self.max_w
        parameters['c_js'] = self.c_js
        parameters['gamma'] = self.gamma
        parameters['P_f'] = self.P_f
        parameters['K_f'] = self.K_f

        with open(path, 'wb') as f:
            pickle.dump(parameters, f)

    def setup_optimizer(self):
        '''Setup the certifying MPC problem.'''

        # Horizon parameter.
        horizon = self.horizon
        nx, nu = self.model.nx, self.model.nu
        # Define optimizer and variables.
        opti = cs.Opti()
        # States.
        z_var = opti.variable(nx, horizon + 1)
        # Inputs.
        v_var = opti.variable(nu, horizon)
        # Lyapunov bound.
        s_var = opti.variable(1, horizon + 1)
        # Error bound.
        w_var = opti.variable(1, horizon + 1)
        # Certified input.
        next_u = opti.variable(nu, 1)
        # Desired input.
        u_L = opti.parameter(nu, 1)
        # Current observed state.
        x_init = opti.parameter(nx, 1)
        # Reference trajectory and predicted LQR gains
        if self.env.TASK == Task.STABILIZATION:
            X_GOAL = opti.parameter(1, nx)
        elif self.env.TASK == Task.TRAJ_TRACKING:
            X_GOAL = opti.parameter(self.horizon, nx)

        for i in range(self.horizon):
            # Dynamics constraints
            next_state = self.dynamics_func(x0=z_var[:, i], p=v_var[:, i])['xf']
            opti.subject_to(z_var[:, i + 1] == next_state)

            # Lyapunov size increase
            s_var[:, i + 1] = self.rho * s_var[:, i] + w_var[:, i]
            opti.subject_to(s_var[:, i] <= self.s_bar_f)
            opti.subject_to(w_var[:, i] <= self.w_bar)
            opti.subject_to(w_var[:, i] >= self.w_func(z_var[:, i], v_var[:, i], s_var[:, i]))

            # Constraints
            for j in range(self.p):
                tighten_by = self.c_js[j] * s_var[:, i + 1]
                opti.subject_to(self.L_x_sym[j, :] @ (z_var[:, i + 1] - self.X_mid) + self.L_u_sym[j, :] @ (v_var[:, i] - self.U_mid) - self.l_sym[j] + tighten_by <= 0)

        # Final state constraints
        if self.use_terminal_set:
            terminal_cost = (z_var[:, -1] - self.X_mid).T @ self.P_f @ (z_var[:, -1] - self.X_mid)
            opti.subject_to(terminal_cost <= self.gamma**2)

        # Initial state constraints
        opti.subject_to(z_var[:, 0] == x_init)
        opti.subject_to(s_var[:, 0] == 0)

        # Real input
        opti.subject_to(next_u == v_var[:, 0])

        # Create solver (IPOPT solver as of this version).
        opts = {'expand': False,
                'ipopt.print_level': 4,
                'ipopt.sb': 'yes',
                'ipopt.max_iter': 50,
                'print_time': 1}
        if self.integration_algo == 'rk4':
            opts['expand'] = True
        opti.solver('ipopt', opts)
        self.opti_dict = {
            'opti': opti,
            'z_var': z_var,
            'v_var': v_var,
            's_var': s_var,
            'w_var': w_var,
            'u_L': u_L,
            'x_init': x_init,
            'next_u': next_u,
            'X_GOAL': X_GOAL,
        }

        # Cost (# eqn 5.a, note: using 2norm or sqrt makes this infeasible).
        cost = self.cost_function.get_cost(self.opti_dict)
        opti.minimize(cost)
