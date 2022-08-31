import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

import ElbowManipulator, ManipulatorController

if __name__ == '__main__' :

	bot = ElbowManipulator.ElbowManipulator()
	controller = ManipulatorController.ManipulatorController(1000)

	bot.setLinkProperties(1, 0.3002, 0.6004)
	bot.setLinkProperties(2, 0.2003, 0.4006)

	controller.system.setLinkProperties(1, 0.3, 0.6)
	controller.system.setLinkProperties(2, 0.2, 0.4)

	# Trajectory
	sim_time = 2
	t = np.linspace(0, sim_time, 10)
	x = np.append(np.linspace(1.0, 0.5, t.shape[0] // 2), np.ones(t.shape[0] // 2) * 0.5)
	y = np.append(np.linspace(0.0, 0.5, t.shape[0] // 2), np.ones(t.shape[0] // 2) * 0.5)

	trajectory = ManipulatorController.Trajectory(t, x, y)
	trajectory.generateTargetStates(controller)

	t_array = list()
	tau1_array = list()
	tau2_array = list()

	def F(t:float, state:ElbowManipulator.ElbowManipulatorState) -> np.ndarray :

		if all(np.isclose(state.q_vec(), trajectory.getTargetState(np.inf)[:2], atol=0.1)) :
			
			return np.array([30, 40])

		else : return np.zeros(2)

	bot.setF(F)

	# Figure
	fig = plt.figure()
	ax, tau_ax = fig.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})

	bot.setUpPlot(ax)
	target_point, = ax.plot([], [], 'o', markersize=3, label='Target Position')

	tau_ax.set_xlim([0, sim_time])
	tau_ax.set_ylim([-20, 20])
	
	tau1_plot, = tau_ax.plot([], [], label=r'$\tau_1$')
	tau2_plot, = tau_ax.plot([], [], label=r'$\tau_2$')

	tau_ax.legend()

	frame_interval = 1 / 60

	def initFrame() :

		global t_array, tau1_array, tau2_array

		bot.setState(0, trajectory.getTargetState(0))
		controller.state.setState(trajectory.getTargetState(0))

		x, y = trajectory.getTargetPosition(0)
		target_point.set_data(x, y)

		t_array = list()
		tau1_array = list()
		tau2_array = list()

		tau1_plot.set_data(t_array, tau1_array)
		tau2_plot.set_data(t_array, tau2_array)

		return bot.updatePlot(), target_point, tau1_plot, tau2_plot,

	def updateFrame(time:float) :

		for t in np.arange(time - frame_interval, time, controller.time_step) :

			controller.state.setState(bot.getState())
			tau = controller.torqueControl(t, trajectory.getTargetState(t), F)

			bot.setTau(tau)

			bot.advanceTime(controller.time_step)
	
		x, y = trajectory.getTargetPosition(time)
		
		target_point.set_data(x, y)

		t_array.append(time)
		tau1_array.append(tau[0])
		tau2_array.append(tau[1])

		tau1_plot.set_data(t_array, tau1_array)
		tau2_plot.set_data(t_array, tau2_array)

		return bot.updatePlot(), target_point, tau1_plot, tau2_plot,

	animation = FuncAnimation(
		fig,
		updateFrame,
		np.arange(0, sim_time, frame_interval) + frame_interval,
		init_func=initFrame,
		blit=True,
		interval = frame_interval * 1000)

	ax.axis('scaled')
	ax.set_axis_off()
	ax.legend()

	plt.show()

	pass