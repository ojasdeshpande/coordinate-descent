import math
from math import *
from random import randint
import numpy as np
from numpy import linalg as LA

class cyclic_coordiante_descent(object):
	
	def __init__(self, n):
		self.eps = 0.00000001
		self.matrix = self.populate_matrix(n)
		self.n = n*4

	# This returns the hessian matrix M for the function in question.
	def populate_matrix(self, n):
		matrix = np.zeros((4*n,4*n)) # [[0 for i in range(4*n)] for j in range(4*n)]
		j = 1
		while(j <= n):
			matrix[0][2*j - 1] = 1.0 / j
			j += 1
		for i in range(0,2*n,2):
			matrix[0][4*n - i - 1] = -1*matrix[0][i+1]
		for i in range(1,4*n):
			for j in range(4*n):
				matrix[i][j] = -1*matrix[i-1][(j+4*n-1)%(4*n)]
		for i in range(4*n):
			matrix[i][i] = 4.5
		return matrix


	def get_ith_matrix(self, coordinate, gamma):
		ith_matrix = np.zeros((self.n, self.n))
		for i in range(self.n):
			ith_matrix[i][i] = 1
		for i in range(self.n):
			ith_matrix[coordinate][i] -= self.matrix[coordinate][i]/gamma
		return ith_matrix

	def calculate_function_value(self, x):
		result = 0.0
		for i in range(n):
			for j in range(i,n):
				if(i != j):
					result += self.matrix[i][j]*x[j]*x[i]
				else:
					result += self.matrix[i][j]*x[i]*x[i]/2
		return result


	def get_norm(self, vector):
		squared_sum = 0
		for i in range(self.n):
			squared_sum += vector[i]*vector[i]
		return math.sqrt(squared_sum)

	def get_next_x(self, x, gamma, coordinate):
		x_new = x[:]
		g_coordinate = 0
		for i in range(n):
			g_coordinate += self.matrix[coordinate][i]*x[i]
		x_new[coordinate] -= g_coordinate/gamma
		return x_new

	def stochastic_coordinate_descent(self, x, gamma):
		steps = 0
		while(1):
			steps += 1
			xx = x[:]
			for _ in range(self.n):
				coordinate = random.randint(0, n-1)
				x = self.get_next_x(x, gamma, coordinate)
			if(abs(self.calculate_function_value(xx)-self.calculate_function_value(x)) < self.eps):
				break
			if((self.calculate_function_value(x)) > ((100*self.eps) + (self.calculate_function_value(xx)))):
				break
		return steps


	def coordinate_descent_with_matrix(self, x, gamma, nfold_matrix):
		steps = 0
		while(1):
			steps += 1
			xx = x[:]
			xx = nfold_matrix.dot(x)
			if(abs(self.calculate_function_value(x)-self.calculate_function_value(xx)) < self.eps):
				break
			if(self.calculate_function_value(xx) > self.calculate_function_value(x) + self.eps*100):
				print("Diverges for gamma", gamma, " and n ", self.n)
				break
			x = xx
		return steps


	def coordinate_descent(self, x, gamma):
		steps = 0
		while(1):
			steps += 1
			xx = x[:]
			for coordinate in range(self.n):
				xx = self.get_next_x(xx, gamma, coordinate)
			if(abs(self.calculate_function_value(x)-self.calculate_function_value(xx)) < self.eps):
				break
			if(self.calculate_function_value(xx) > self.calculate_function_value(x) + self.eps*100):
				print("Diverges for gamma", gamma, " and n ", self.n)
				break
			x = xx
		return steps

	def shift_matrix(self, matrix, steps):
		cur = np.zeros((self.n, self.n))
		for i in range(self.n):
			for j in range(self.n):
				cur[i][j] = matrix[(i-steps+self.n)%self.n][(j-steps+self.n)%self.n]
		return cur

	def get_n_fold_matrix_product(self, gamma):
		cur = self.get_ith_matrix(0, gamma)
		cur = np.matmul(self.get_ith_matrix(1, gamma), cur)
		steps = 2
		while(steps < self.n):
			# print(steps)
			cur = np.matmul(self.shift_matrix(cur, steps), cur)
			steps *= 2
		return cur
		for i in range(1, self.n):
			# print(i)
			cur = np.matmul(self.get_ith_matrix(i, gamma), cur)
		return cur

	def get_n_fold_matrix_product_naive(self, gamma):
		cur = self.get_ith_matrix(0, gamma)
		for i in range(1, self.n):
			cur = np.matmul(self.get_ith_matrix(i, gamma), cur)
		return cur
		# print(cur[0])

if __name__ == "__main__":
	
	for nn in range(3, 11):
		n = 2**nn
		ccd = cyclic_coordiante_descent(n)
		print("N = ", n)
		for step_size in range(20, 90):
			results = []
			gamma = step_size / 10
			print(gamma)
			nfold_matrix = ccd.get_n_fold_matrix_product(gamma)
			w, v = LA.eig(nfold_matrix)
			worst = 0
			worst_vec = []
			worst_eig = 0
			for i in range(len(w)):
				eigenvalue = w[i]
				if np.absolute(eigenvalue) > worst:
					worst = np.absolute(eigenvalue)
					worst_vec = v[i]
					worst_eig = eigenvalue
			# select the eigenvector corresponding to the eigenvalue with the maximum magnitude
			steps = ccd.coordinate_descent_with_matrix(worst_vec.real, gamma, nfold_matrix)
			results.append(steps)
			steps = ccd.coordinate_descent(worst_vec.real, gamma)
			# ideally coordinate descent with the matrix and along with each individual coordinate
			# should give the same answer.
			print("random vectors now")
			results.append(steps)
			for j in range(5):
				x = [randint(-1, 1) for _ in range(4*n)]
				results.append(ccd.coordinate_descent(x, gamma))
			print(results)


