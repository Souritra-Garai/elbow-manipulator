import numpy as np

g = 9.8	# m2 / s

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
		
		self._q 		= np.zeros(2)	# radians
		self._q_dot		= np.zeros(2)	# rad / s
		self._q_ddot	= np.zeros(2)	# rad / s2

		self._F			= np.zeros(2)	# N
		self._tau		= np.zeros(2)	# N m
		
		pass

	def getJ1(self) -> np.ndarray :

		return np.zeros(2)

	def getLinkEnd(self, index = 0) -> np.ndarray :

		return self._l[index] * np.array([
			np.cos(self._q[index]),
			np.sin(self._q[index])
		])

	def getJ2(self) -> np.ndarray :

		return self.getLinkEnd(0)

	def getE(self) -> np.ndarray :

		return self.getJ2() + self.getLinkEnd(1)

	def getJacobian(self) -> np.ndarray :

		return np.array([
			[-	self._l[0] * np.sin(self._q[0]), -	self._l[1] * np.sin(self._q[1])],
			[	self._l[0] * np.cos(self._q[0]),	self._l[1] * np.cos(self._q[1])]
		])


def __det(mat:np.ndarray) -> float :

	return mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]


