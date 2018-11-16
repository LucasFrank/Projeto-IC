# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import random
import pandas as pd
import csv
import time
import math
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

def MAPE(y_true, y_pred):
	errors = 0
	for i in range(len(y_true)):
		errors = errors + (abs((float(y_true[i]) - float(y_pred[i]))) / float(y_true[i]))
	return errors / len(y_pred)

def modelNN(n,shape,epochs,learning_rate,alpha):
	cf = 'custom_activation'
	model = Sequential()

	get_custom_objects().update({cf: Activation(custom_function(alpha))})
	model.add(Dense(units = n,input_shape = shape ,activation = cf, kernel_initializer='normal'))
	model.add(Dense(units = n, activation = cf, kernel_initializer = 'normal'))
	model.add(Dense(units = n, activation = cf, kernel_initializer = 'normal'))
	model.add(Dense(units = n, activation = cf, kernel_initializer = 'normal'))
	model.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'normal'))
	decay_rate = learning_rate/epochs;
	adam = Adam(lr = learning_rate,decay = decay_rate)
	model.compile(loss='mean_squared_error', optimizer=adam)
	return model

def custom_function(alpha):
	def custom_activation(x):
		return 1 / (1 + K.exp(-alpha * x))
	
	return custom_activation

class VariablesControl:
	
	# Variaveis PSO
	c1 = 0.6
	c2 = 0.8
	w = 1
	g_best_pos = []
	g_best_cost = 100

	# Posição Limite
	n_MAX = 120
	n_MIN = 60
	alpha_MAX = 52
	alpha_MIN = 2
	learning_rate_MAX = 0.001
	learning_rate_MIN = 0.0001

	# Velocidade Limite
	n_velocity_MAX = 0.1 * (n_MAX - n_MIN)
	n_velocity_MIN = -0.1 * (n_MAX - n_MIN)
	alpha_velocity_MAX = 0.1 * (alpha_MAX - alpha_MIN)
	alpha_velocity_MIN = -0.1 * (alpha_MAX - alpha_MIN)
	learning_rate_velocity_MAX = 0.1 * (learning_rate_MAX - learning_rate_MIN)
	learning_rate_velocity_MIN = -0.1 * (learning_rate_MAX - learning_rate_MIN)
	
	
class Particle(VariablesControl):
	
	def __init__(self,pos,vel,cost):
		self.num_of_param = 3
		self.p_best_pos = pos
		self.position = pos
		self.velocity = vel
		self.p_best_cost = cost
		
		if cost < VariablesControl.g_best_cost:
			VariablesControl.g_best_cost = cost
			VariablesControl.g_best_pos = pos.copy()
		
		
	def calculatePosition(self):
		for i in range(self.num_of_param):
			self.position[i] = self.position[i] + self.velocity[i]
			if i == 0:
				self.position[i] = max(self.position[i],self.n_MIN)
				self.position[i] = min(self.position[i],self.n_MAX)
			elif i == 1:
				self.position[i] = max(self.position[i],self.alpha_MIN)
				self.position[i] = min(self.position[i],self.alpha_MAX)
			elif i == 2:
				self.position[i] = max(self.position[i],self.learning_rate_MIN)
				self.position[i] = min(self.position[i],self.learning_rate_MAX)
		
	def calculateVelocity(self):
		for i in range(self.num_of_param):
			self.velocity[i] = self.w * self.velocity[i] + self.c1 * random.uniform(0,1) * (self.p_best_pos[i] - self.position[i]) + self.c2 * random.uniform(0,1) * (self.g_best_pos[i] - self.position[i])
			if i == 0:
				self.velocity[i] = max(self.velocity[i],self.n_velocity_MIN)
				self.velocity[i] = min(self.velocity[i],self.n_velocity_MAX)
			elif i == 1:
				self.velocity[i] = max(self.velocity[i],self.alpha_velocity_MIN)
				self.velocity[i] = min(self.velocity[i],self.alpha_velocity_MAX)
			elif i == 2:
				self.velocity[i] = max(self.velocity[i],self.learning_rate_velocity_MIN)
				self.velocity[i] = min(self.velocity[i],self.learning_rate_velocity_MAX)
			
	def setGBest(self, position,value):
		VariablesControl.g_best_pos = position.copy()
		VariablesControl.g_best_cost = value;
		
	def setPBest(self,position,value):
		self.p_best_pos = pos.copy()
		self.p_best_cost = value
		

	


# Variaveis Fixadas
MAX_Q = 5
p = 13
q = 4
epochsN = 1 # modificar talvez

# Variaveis a ser otimizadas
n = 100
learning_rate = 0.005
alpha = 1

# Variaveis PSO
num_of_iterations = 10
population_size = 10

# Loading Data
df = pd.read_csv("pems.csv", header=0)

minMapeMPL1 = 100.0

# Indexing the data
df['date'] = pd.to_datetime(df['date'])
df.index = df['date']
del df['date']

with open('Results/NN_PEMS.csv', 'w', 1) as nn_file:
	# Reading CSV
	nnwriter  = csv.writer(nn_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)

	# Writing results headers
	nnwriter.writerow(['P', 'Q', 'N', 'H', 'Avg Mape', 'Min MAPE', 'Avg time'])

	# Using historic data (Q) from the same time and weekday
	for i in range (1, MAX_Q + 1):
		df['count-{}'.format(i)] = df['count'].shift(i * 7 * 24)

	# Change n/a to 1
	df = df.fillna(0)

	# Normalizing the data
	df_max = max(df['count'])
	df_min = min(df['count'])
	df['count'] = df['count'] / (df_max - df_min)
	for i in range (1, MAX_Q + 1):
		df['count-{}'.format(i)] = df['count-{}'.format(i)] / (df_max - df_min)

	aux_df = df

	# Shifiting the data set by Q weeks
	df = df[q * 7 * 24:]

	print('Running for params P = {}, Q = {}, N = {}'.format(p, q, n))
	print('Pre-processing...')

		# Initializing the data
	X1 = list()
	Y1 = list()

	# Mapping each set of variables (P and Q) to their correspondent value
	for i in range(len(df) - p - 1):
		X = list()
		for j in range (1, q + 1):
			X.append(df['count-{}'.format(j)][i + p + 1])

		X1.append(X + list(df['count'][i:(i + p)]))
		Y1.append(df['count'][i + p + 1])

	print('	  Splitting in train-test...')
		# Train/test/validation split
	rows1 = random.sample(range(len(X1)), int(len(X1)//3))
	
	X1_test = np.array( [X1[j] for j in rows1] )
	Y1_test = np.array( [Y1[j] for j in rows1] )
	X1_train = np.array( [X1[j] for j in list(set(range(len(X1))) - set(rows1))] )
	Y1_train = np.array([Y1[j] for j in list(set(range(len(Y1))) - set(rows1))] )
	
	print('	  Initializing the models...')
	results_nn1 = list()
	avg_mlp_time1 = 0
	# Initializing the model
	shape = X1_train.shape[1:]
	
	# Initializing the variables and the population
	vc = VariablesControl
	pop = []
	for i in range(population_size):
		pos = []
		n = random.randrange(vc.n_MIN,vc.n_MAX) # 0
		alpha = random.randrange(vc.alpha_MIN,vc.alpha_MAX) # 1
		learning_rate = random.uniform(vc.learning_rate_MIN,vc.learning_rate_MAX) # 2
		pos.append(n); pos.append(alpha); pos.append(learning_rate)
		
		vel = []
		n_v = random.randrange(vc.n_velocity_MIN,vc.n_velocity_MAX)
		alpha_v = random.randrange(vc.alpha_velocity_MIN,vc.alpha_velocity_MAX)
		learning_rate_v = random.uniform(vc.learning_rate_velocity_MIN,vc.learning_rate_velocity_MAX)
		vel.append(n); vel.append(alpha); vel.append(learning_rate)
		
		MLP1 = modelNN(n,shape,epochsN,learning_rate,alpha)
		MLP1.fit(X1_train, Y1_train, epochs = epochsN)
		predicted1_nn = MLP1.predict(X1_test)
		cost = MAPE(Y1_test, predicted1_nn)
		
		p = Particle(pos,vel,cost)
		pop.append(p)
	
	iteration = 0
	while(iteration < num_of_iterations):
		num_particle = 0
		for particle in pop:
			particle.calculateVelocity()
			particle.calculatePosition()
			
			n = int(particle.position[0])
			learning_rate = particle.position[2]
			alpha = particle.position[1]
			MLP1 = modelNN(n,shape,epochsN,learning_rate,alpha)

			print('Running tests...')
			for test in range(0, 30):
				if(test % 6 == 5):
					print('T = {}%'.format(int(((test + 1)*100)/30)))

				start_time = time.time()
				
				MLP1.fit(X1_train, Y1_train, epochs = epochsN)
				predicted1_nn = MLP1.predict(X1_test)
				currentCost = MAPE(Y1_test, predicted1_nn) 
				
				avg_mlp_time1 = avg_mlp_time1 + time.time() - start_time
				
				results_nn1.append(MAPE(Y1_test, predicted1_nn))
				if(minMapeMPL1 > MAPE(Y1_test, predicted1_nn)):
					trueValue = pd.DataFrame(Y1_test)
					bestMLP1value = pd.DataFrame(predicted1_nn)
					minMapeMPL = MAPE(Y1_test, predicted1_nn)
					trueValue.to_csv("Results/TrueValue.csv")
					bestMLP1value.to_csv("Results/BestMLP1value.csv")
				
			if currentCost < particle.p_best_cost:
				particle.setPBest(particle.position,currentCost)
				
				if particle.p_best_cost < particle.g_best_cost:
					particle.setGBest(particle.p_best_pos,particle.p_best_cost)
			
			print("Particle {}.".format(num_particle))
			num_particle += 1;
		
		# print the best pos and cost of the population so far
		print("Position = {}".format(pop[0].g_best_pos))
		print("Cost = {}".format(pop[0].g_best_cost))
		iteration += 1


	nnwriter.writerow([p, q, n, 1, np.mean(results_nn1), min(results_nn1), avg_mlp_time1 / 30])
	print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
df = aux_df
