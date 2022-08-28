from operator import index
from textwrap import indent
import numpy as np
import ElbowManipulator

from scipy.interpolate import interp1d

class ManipulatorController() :

	def __init__(self, frequency = 1000) -> None:
		
		self.state	= ElbowManipulator.ElbowManipulatorState()
		self.system	= ElbowManipulator.ElbowManipulatorConfig()

		self.__J = self.state.J(self.system)
		self.time_step = 1 / frequency

		pass

	# Inverse Kinematics

	def q(self, x:np.ndarray, y:np.ndarray) -> np.ndarray :

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

		return np.column_stack((
			phi + varphi,
			phi + varphi - theta
		))


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

		t = np.append(t, np.inf)
		x = np.append(x, x[-1])
		y = np.append(y, y[-1])
		
		q1 = np.append(q1, q1[-1])
		q2 = np.append(q2, q2[-1])

		self.__t = t
		self.__x = x
		self.__y = y

		q1_dot = (q1[1:] - q1[:-1]) / (t[1:] - t[:-1])
		q2_dot = (q2[1:] - q2[:-1]) / (t[1:] - t[:-1])

		q1_dot = np.append(q1_dot, 0)
		q2_dot = np.append(q2_dot, 0)

		self.__target_states = np.column_stack((q1, q2, q1_dot, q2_dot))
		
		pass
	
	def getIndex(self, t) -> int :

		i = np.argmin(np.abs(self.__t - t))
		i += 1 * (self.__t[i] < t)

		return i

	def getTargetState(self, t) -> np.ndarray :

		return self.__target_states[self.getIndex(t)]

	def getTargetPosition(self, t) -> np.ndarray :

		i = self.getIndex(t)

		return self.__x[i], self.__y[i]

	def n(self) :

		return self.__t.shape[0]

if __name__ == '__main__' :

	bot = ElbowManipulator.ElbowManipulator()
	controller = ManipulatorController(1000)

	t = np.linspace(0, 10, 50)
	x = 0.5 * np.cos(1 * t * np.pi)
	y = 0.5 + 0.5 * np.sin(1 * t  * np.pi)

	trajectory = Trajectory(t, x, y)
	
	trajectory.generateTargetStates(controller)
	
	import matplotlib.pyplot as plt
	from matplotlib.animation import FuncAnimation

	fig = plt.figure()

	ax = fig.add_subplot(1, 1, 1)

	ax.axis('scaled')

	bot.setUpPlot(ax)

	target, = plt.plot([], [], 'o', markersize=10)

	ax.set_axis_off()

	interval = 100 // 144

	def update(i) :

		t = i * controller.time_step

		bot.setState(t, trajectory.getTargetState(t))

		x, y = trajectory.getTargetPosition(t)
		
		target.set_data(x, y)

		return bot.updatePlot(), target

	my_anim = FuncAnimation(fig, update, np.arange(start=0, stop=trajectory.n(), step=1), blit=True, interval = interval)

	# plt.show()

	plt.show()

	pass