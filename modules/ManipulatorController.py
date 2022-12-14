import numpy as np
import modules.ElbowManipulator as ElbowManipulator

class ManipulatorController() :

	def __init__(self, frequency = 1000) -> None:
		
		self.state	= ElbowManipulator.ElbowManipulatorState()
		self.system	= ElbowManipulator.ElbowManipulatorConfig()

		self.__J = self.state.J(self.system)
		self.time_step = 1 / frequency

		pass

	def angleControl(self, t:float, target_state:np.ndarray) -> np.ndarray :

		return target_state

	def torqueControl(self, t:float, target_state:np.ndarray, F:callable) -> np.ndarray :

		t_state = ElbowManipulator.ElbowManipulatorState()
		t_state.setState(target_state)

		tau  = self.state.getConfigTerms(self.system)
		tau += self.state.getGravityTerms(self.system) 
		tau += np.matmul(self.state.J(self.system), F(t, t_state.E(self.system)))
		
		tau += np.matmul(self.state.getAccMatrix(self.system), (t_state.q_dot_vec() - self.state.q_dot_vec()) / self.time_step)

		return tau

	# Inverse Kinematics

	def q(self, x:np.ndarray, y:np.ndarray) -> np.ndarray :

		if np.any(x**2 + y**2 > (self.system.l(1) + self.system.l(2))**2) :

			raise ValueError('Desired position out of system bounds')

		theta = np.arccos(
			(x**2 + y**2 - self.system.l(1)**2 - self.system.l(2)**2) /
			(2.0 * self.system.l(1) * self.system.l(2))
		)

		theta *= 1 - 2 * (np.signbit(x))

		phi		= np.arctan2(y, x)
		varphi	= np.arctan2(
			 self.system.l(2) * np.sin(theta),
			(self.system.l(1) + self.system.l(2) * np.cos(theta))
		)

		q_vec = np.column_stack((
			phi + varphi,
			phi + varphi - theta
		))

		# if not self.system.withinSystemBounds(q_vec) :

		# 	print(q_vec)
		# 	raise ValueError('Desired position out of system bounds')

		return q_vec

class Trajectory() :

	def __init__(self, t:np.ndarray, x:np.ndarray, y:np.ndarray) -> None :

		if len(t) == len(x) == len(y) :
			
			if np.all(t[1:] > t[:-1]) :
				
				self.__t = t
				self.__x = x
				self.__y = y

			else : raise ValueError('Time array must be monotonically increasing')

		else : raise TypeError('All arrays should be of same length')

		pass

	def generateTargetStates(self, controller:ManipulatorController) -> None :

		q = controller.q(self.__x, self.__y)

		n = int(np.ceil(self.__t[-1] / controller.time_step)) + 1

		t = np.zeros(0)
		x = np.zeros(0)
		y = np.zeros(0)

		q1 = np.zeros(0)
		q2 = np.zeros(0)

		f = lambda x, x0, x1, y0, y1 : y0 + (y1 - y0) * (x - x0) / (x1 - x0)

		for i in range(1, len(self.__t)) :

			t_arr = np.arange(self.__t[i-1], self.__t[i], controller.time_step)
			
			n = t_arr.shape[0]

			t = np.append(t, t_arr)
			x = np.append(x, np.linspace(self.__x[i-1], f(t_arr[-1], t_arr[0], self.__t[i], self.__x[i-1], self.__x[i]), n))
			y = np.append(y, np.linspace(self.__y[i-1], f(t_arr[-1], t_arr[0], self.__t[i], self.__y[i-1], self.__y[i]), n))
			
			q1 = np.append(q1, np.linspace(q[i-1, 0], q[i, 0], n))
			q2 = np.append(q2, np.linspace(q[i-1, 1], q[i, 1], n))

		# t = np.append(t, np.inf)
		# x = np.append(x, x[-1])
		# y = np.append(y, y[-1])
		
		# q1 = np.append(q1, q1[-1])
		# q2 = np.append(q2, q2[-1])

		self.__t = t
		self.__x = x
		self.__y = y

		q1_dot = (q1[1:] - q1[:-1]) / (t[1:] - t[:-1])
		q2_dot = (q2[1:] - q2[:-1]) / (t[1:] - t[:-1])

		q1_dot = np.append(q1_dot, q1_dot[-1])
		q2_dot = np.append(q2_dot, q2_dot[-1])

		self.__target_states = np.column_stack((q1, q2, q1_dot, q2_dot))
		
		pass
	
	def getIndex(self, t:float) -> int :

		i = np.argmin(np.abs(self.__t - t))
		i -= 1 * (self.__t[i] > t)

		return i

	def getTargetState(self, t:float) -> np.ndarray :

		return self.__target_states[self.getIndex(t)]

	def getTargetPosition(self, t:float) -> np.ndarray :

		i = self.getIndex(t)

		return self.__x[i], self.__y[i]

	def n(self) -> int :

		return self.__t.shape[0]

if __name__ == '__main__' :

	bot = ElbowManipulator.ElbowManipulator()
	controller = ManipulatorController(1000)

	t = np.linspace(0, 10, 200)
	x = 0.5 * np.cos(2 * t * np.pi)
	y = 1 + 0.5 * np.sin(2 * t  * np.pi)

	trajectory = Trajectory(t, x, y)
	
	trajectory.generateTargetStates(controller)

	bot.setState(0, trajectory.getTargetState(0))
	
	import matplotlib.pyplot as plt
	from matplotlib.animation import FuncAnimation

	fig = plt.figure()

	ax = fig.add_subplot(1, 1, 1)

	ax.axis('scaled')

	bot.setUpPlot(ax)

	target, = plt.plot([], [], 'o', markersize=10)

	ax.set_axis_off()

	interval = 1 / 144

	F = lambda t, state : np.zeros(2)

	def update(time) :

		for t in np.arange(time - interval, time, controller.time_step) :

			controller.state.setQ(bot.getQ())
			controller.state.setQdot(bot.getQdot())

			bot.setTau(controller.torqueControl(t, trajectory.getTargetState(t), F))

			bot.advanceTime(controller.time_step)

		x, y = trajectory.getTargetPosition(t)
		
		target.set_data(x, y)

		return bot.updatePlot(), target

	my_anim = FuncAnimation(fig, update, np.arange(0, 10, interval), blit=True, interval = interval * 1000)

	# plt.show()

	plt.show()

	pass