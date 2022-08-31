import numpy as np

from scipy.integrate import solve_ivp
from matplotlib.pyplot import Axes, Line2D

g = 9.8	# m2 / s

class ElbowManipulatorConfig :

	def __init__(self) -> None:

		self._l	= np.ones(2)	# m
		self._m	= np.ones(2)	# kg
		
		# Absolute angle limits [min, max]
		# on joint angles [q1, q2]
		self._q_lim	= np.pi * np.array([
			[ 0,	1  ],
			[-0.5,	1.5]
		])
		
		pass

	# Get / Set properties

	def setLinkProperties(self, link_num:int, mass:float, length:float) -> None :

		if link_num not in [1, 2] : raise ValueError("Link number must be 1 or 2")

		link_num -= 1

		if mass > 0 : self._m[link_num] = mass
		else : raise ValueError("Mass must be positive")
		
		if length > 0 : self._l[link_num] = length
		else : raise ValueError("Length must be positive")
		
		pass

	def setJointLimits(self, joint_num:int, min_angle:float, max_angle:float) -> None :

		if joint_num not in [1, 2] : raise ValueError("Link number must be 1 or 2")

		joint_num -= 1

		if min_angle >= 0 and min_angle <= max_angle : self._q_lim[joint_num, 0] = min_angle
		else : raise ValueError("min_angle must lie in the range [0, max_angle]")

		if max_angle >= min_angle and max_angle <= np.pi : self._q_lim[joint_num, 1] = max_angle
		else : raise ValueError("max_angle must lie in the range [min_angle, pi]")

		pass

	def m(self, i:int) -> float :

		return self._m[i-1]

	def l(self, i:int) -> float :

		return self._l[i-1]

	def m_vec(self) -> np.ndarray :

		return np.array(self._m)

	def l_vec(self) -> np.ndarray :

		return np.array(self._l)

	# System checks

	def withinSystemBounds(self, q_vec:np.ndarray) -> bool :

		ret_val  = np.all(np.bitwise_and(self._q_lim[:, 0] <= q_vec, q_vec <= self._q_lim[:, 1]))
		ret_val *= np.all(sum(self.l_vec() * np.sin(q_vec)) >= 0)

		return ret_val

class ElbowManipulatorState(np.ndarray) :

	def __new__(subtype) -> None:
		
		obj = super().__new__(subtype, shape=(4), dtype=float)

		obj.setState(np.zeros(4))

		return obj

	def __array_finalize__(self, obj) -> None:

		if obj is None : return
		
		elif self.shape[-1] != 4 : raise TypeError('Array size should be 4')

		pass

	# Get / Set attributes

	def q(self, i:int) -> float :

		return self[i-1]

	def q_dot(self, i:int) -> float :

		return self[i+1]

	def q_vec(self) -> np.ndarray :

		return np.array([self[0], self[1]])

	def q_dot_vec(self) -> np.ndarray :

		return np.array([self[2], self[3]])

	def setQ(self, q:np.ndarray) -> None :

		self[0] = q[0]
		self[1] = q[1]

		pass

	def setQdot(self, q_dot:np.ndarray) -> None :

		self[2] = q_dot[0]
		self[3] = q_dot[1]

		pass

	def setState(self, Q:np.ndarray) -> None :

		self[:] = Q[0:4]

		pass

	# Utility matrices and vectors

	def J(self, system:ElbowManipulatorConfig) -> np.ndarray :

		return np.array([
			[	- system.l(1) * np.sin(self.q(1)),	- system.l(2) * np.sin(self.q(2))],
			[	  system.l(1) * np.cos(self.q(1)),	  system.l(2) * np.cos(self.q(2))]
		])

	def getAccMatrix(self, system:ElbowManipulatorConfig) -> np.ndarray :

		config_term = system.m(2) * system.l(1) * 0.5 * system.l(2) * np.cos(self.q(1) - self.q(2))

		return np.array([
			[	(system.m(1) / 3.0 + system.m(2)) * (system.l(1)**2),	config_term],
			[	config_term,											system.m(2) * (system.l(2)**2) / 3.0]
		])

	def getConfigTerms(self, system:ElbowManipulatorConfig) -> np.ndarray :

		config_term = system.m(2) * system.l(1) * 0.5 * system.l(2) * np.sin(self.q(1) - self.q(2))
		config_term = config_term * np.array([
			- self.q_dot(2)**2,
			  self.q_dot(1)**2
		])

		return config_term

	def getGravityTerms(self, system:ElbowManipulatorConfig) -> np.ndarray :

		return system.m_vec() * g * 0.5 * system.l_vec() * np.cos(self.q_vec()) + np.array([
			system.m(2) * g * system.l(1) * np.cos(self.q(1)),
			0
		])

	# Forward Kinematics

	def E(self, system:ElbowManipulatorConfig) -> np.ndarray :

		return np.append(
			np.array([
				system.l(1) * np.cos(self.q(1)) + system.l(2) * np.cos(self.q(2)),
				system.l(1) * np.sin(self.q(1)) + system.l(2) * np.sin(self.q(2))
			]),
			np.matmul(self.J(system), self.q_dot_vec())
		)

	# Dynamics

	def q_ddot(self, system:ElbowManipulatorConfig, tau:np.ndarray, F:np.ndarray) -> np.ndarray :

		return np.matmul(
			_inv(self.getAccMatrix(system)),
			tau - np.matmul(self.J(system).T, F) - self.getConfigTerms(system) - self.getGravityTerms(system)
		)

class ElbowManipulator() :

	def __init__(self) -> None:
		
		self.__system	= ElbowManipulatorConfig()
		self.__state	= ElbowManipulatorState()

		self.__tau		= np.zeros(2)
		self.__F		= lambda t, state : np.zeros(2)
		self.__t		= 0.0
		
		pass

	# Dynamics

	def __getDerivative(self, t:float, y:np.ndarray) -> np.ndarray :

		state = ElbowManipulatorState()
		state.setState(y)

		return np.append(
			state.q_dot_vec(),
			state.q_ddot(self.__system, self.__tau, self.__F(t, state.E(self.__system)))
		)
	
	def advanceTime(self, time_step:float) -> None :

		solution = solve_ivp(self.__getDerivative, (self.__t, self.__t + time_step), self.__state)

		if solution.success :

			self.setState(solution.t[-1], solution.y[:, -1])

		else : 

			raise RuntimeError('Numerical solution to governing differential equations did not converge')	
		
		pass

	# Get / Set attributes

	def getQ(self) -> np.ndarray :

		return self.__state.q_vec()

	def getQdot(self) -> np.ndarray :

		return self.__state.q_dot_vec()
	
	def getTime(self) -> float :

		return self.__t
	
	def getState(self) -> ElbowManipulatorState :

		return np.copy(self.__state)

	def setF(self, F:callable) -> None :

		if F(self.__t, self.__state).shape[-1] == 2 :

			self.__F = F

		else :

			raise ValueError('Method F(t, state) should take two arguments and output [...,2] numpy.ndarray')

	def setTau(self, tau:np.ndarray) -> None :

		self.__tau[:] = tau

	def setState(self, t, state:np.ndarray) -> None :

		if self.__system.withinSystemBounds(state[:2]) :

			self.__t = t
			self.__state.setState(state)

		else :

			raise ValueError('State out of system bounds')

		pass

	def setLinkProperties(self, link_num:int, mass:float, length:float) -> None :

		if self.__t == 0 : self.__system.setLinkProperties(link_num, mass, length)
		else : raise RuntimeError('Simulation time not zero')
		pass

	def setJointLimits(self, joint_num:int, min_angle:float, max_angle:float) -> None :

		if self.__t == 0 : self.__system.setJointLimits(joint_num, min_angle, max_angle)
		else : raise RuntimeError('Simulation time not zero')
		pass

	# Plot

	def setUpPlot(self, axes:Axes) -> tuple :

		self.__plot_obj, = axes.plot([], [], 'o-', lw=5)

		l = sum(self.__system.l_vec())
		axes.set_xlim([-1.5  * l, 1.5 * l])
		axes.set_ylim([-0.25 * l, 1.5 * l])

		return self.__plot_obj

	def updatePlot(self) -> tuple :

		x = np.zeros(3)
		y = np.zeros(3)

		x[1:] = self.__system.l_vec() * np.cos(self.__state.q_vec())
		y[1:] = self.__system.l_vec() * np.sin(self.__state.q_vec())

		x[2] += x[1]
		y[2] += y[1]

		self.__plot_obj.set_data(x, y)

		return self.__plot_obj

def _det(mat:np.ndarray) -> float :

	assert(mat.shape == (2,2))

	return mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]

def _inv(mat:np.ndarray) -> np.ndarray :

	return np.array([
		[ mat[1,1],	-mat[0,1]],
		[-mat[1,0],	 mat[0,0]]
	]) / _det(mat)

if __name__ == '__main__' :

	bot = ElbowManipulator()
	bot.setState(0, np.array([np.pi/3, np.pi / 3, 0, 0]))
	bot.setTau([1.5 * g * np.cos(np.pi/3), 0.5 * g * np.cos(np.pi/4)])

	import matplotlib.pyplot as plt
	from matplotlib.animation import FuncAnimation

	fig = plt.figure()

	ax = fig.add_subplot(1, 1, 1)

	bot.setUpPlot(ax)

	ax.set_axis_off()

	# title = ax.text(0.5,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
	#             transform=ax.transAxes, ha="center", s=0)

	interval = 1000 // 144

	def update(i) :

		bot.advanceTime(interval / 1000)

		return bot.updatePlot(),

	my_anim = FuncAnimation(fig, update, np.arange(start=0, stop=100, step=1), blit=True, interval = interval)

	# plt.show()

	plt.show()

	pass