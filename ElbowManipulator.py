import numpy as np

from scipy.integrate import solve_ivp

g = 9.8	# m2 / s

__time_step = 1E-4	# s

class ElbowManipulatorConfig :

	def __init__(self) -> None:
		
		self._l	= np.ones(2)	# m
		self._m	= np.ones(2)	# kg
		
		# Absolute angle limits [min, max]
		# on joint angles [q1, q2]
		self._q_lim	= np.zeros((2, 2))	# radians
		self._q_lim[:, 1] = np.pi
		
		pass

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

class ElbowManipulator(ElbowManipulatorConfig) :

	def __init__(self) -> None:

		self._t			= 0.0			# s
		
		self._q 		= np.zeros(2)	# radians
		self._q_dot		= np.zeros(2)	# rad / s
		self._q_ddot	= np.zeros(2)	# rad / s2

		self._J			= self.getJacobian()

		self._F			= np.zeros(2)	# N
		self._tau		= np.zeros(2)	# N m
		
		pass

	# Utility vector and matrices

	def getLinkEnd(self, index = 0) -> np.ndarray :

		return self._l[index] * np.array([
			np.cos(self._q[index]),
			np.sin(self._q[index])
		])

	def getJacobian(self) -> np.ndarray :

		return np.array([
			[-	self._l[0] * np.sin(self._q[0]), -	self._l[1] * np.sin(self._q[1])],
			[	self._l[0] * np.cos(self._q[0]),	self._l[1] * np.cos(self._q[1])]
		])

	def getConfigAcc(self) -> np.ndarray :
		
		return np.array([
			np.sum(self._l * np.cos(self._q) * (self._q_dot ** 2)),
			np.sum(self._l * np.sin(self._q) * (self._q_dot ** 2))
		])

	def getAccMatrix(self) -> np.ndarray :

		config_term = self._m[1] * self._l[0] * self._l[1] * np.cos(self._q[0] - self._q[1]) / 2

		return np.array([
			[	((self._m[0] / 3) + self._m[1]) * (self._l[0]**2),	config_term],
			[	  self._m[1] * (self._l[1]**2) / 3,					config_term]
		])

	def getConfigForce(self) -> np.ndarray :

		config_term = self._m[1] * self._l[0] * self._l[1] * np.sin(self._q[0] - self._q[1]) / 2

		gravity_term = self._m * g * self._l * np.cos(self._q) / 2 + np.array([
			self._m[1] * g * self._l[0] * np.cos(self._q[0])
		])

		return config_term * (self._q_dot**2) * np.array([1, -1]) + gravity_term

	def __getDerivative(self, t:float, Q:np.ndarray, tau:np.ndarray, F:callable) :

		self._t			= t
		self._q[:]		= Q[0:2]
		self._q_dot[:]	= Q[2:4]
		self._q_ddot[:]	= self.getQDDot(tau, F)

		return np.append(
			self._q_dot,
			self._q_ddot
		)

	# Forward Kinematics

	def getJ1(self) -> np.ndarray :

		return np.zeros(2)

	def getJ2(self) -> np.ndarray :

		return self.getLinkEnd(0)

	def getE(self) -> np.ndarray :

		return self.getJ2() + self.getLinkEnd(1)

	def getEVel(self) -> np.ndarray :

		return np.matmul(self._J, self._q_dot)
	
	def getEAcc(self) -> np.ndarray :

		return np.matmul(self._J, self._q_ddot) - self.getConfigAcc()

	# Inverse Kinematics

	def getQ(self, pos_E:np.ndarray=np.zeros(2)) -> np.ndarray :

		theta	= np.arccos(np.sum(pos_E**2 - self._l**2) / (2 * self._l[0] * self._l[1]))
		theta	*= 1 - 2 * np.signbit(pos_E[0])

		phi		= np.arctan2(pos_E[1], pos_E[0])
		varphi	= np.arctan2(
			self._l[1] * np.sin(theta),
			self._l[0] + self._l[1] * np.cos(theta)
		)

		return np.array([
			phi + varphi,
			phi + varphi - theta
		])

	def getQDot(self, vel_E:np.ndarray=np.zeros(2)) -> np.ndarray :

		return np.matmul(__inv(self._J), vel_E)

	def getQDDot(self, acc_E:np.ndarray=np.zeros(2)) -> np.ndarray :

		return np.matmul(
			__inv(self._J),
			acc_E + self.getConfigAcc()
		)

	# Dynamics

	def getQDDot(self, tau:np.ndarray, F:callable) -> np.ndarray :

		return np.matmul(
			__inv(self.getAccMatrix()),
			tau + np.matmul(
				- np.transpose(self._J),
				F(self._t, self._q, self._q_dot, self._q_ddot)
			)
		)

	def getDerivativeLagrangian(self) -> np.ndarray :

		return np.matmul(self.getAccMatrix(), self._q_ddot) + self.getConfigForce()

	def advanceTimeStep(self, time_step, tau:np.ndarray, F:callable) -> None :

		solve_ivp(
			self.__getDerivative,
			(self._t, self._t + time_step),
			np.append(self._q, self._q_dot),
			args=(tau, F),
			max_step = __time_step
		)

		pass

def __det(mat:np.ndarray) -> float :

	return mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]

def __inv(mat:np.ndarray) -> np.ndarray:

	return np.array([
		[ mat[1,1],	-mat[0,1]],
		[-mat[1,0],	 mat[0,0]]
	]) / __det(mat)


if __name__ == '__main__' :

	pass