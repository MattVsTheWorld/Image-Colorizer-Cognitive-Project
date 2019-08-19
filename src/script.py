%cpaste

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.interactive(True)
import cv2,os
from scipy import stats
from skimage import color
import tensorflow as tf
import random

# Clears the default graph stack and resets the global default graph.
tf.reset_default_graph()

# ?
eps=10e-9
# Distributed map showing where most of the color values are found, loaded from file
DistrMap = np.load("distrMap_all.npy") # 256x256
map_size = DistrMap.shape[0]
# Width and height of the window used to divide the distrMap equally
WinMap = 8 # 256/8 -> 32. 32*32 = 1024 tiles.
# Small map: contains the tiles already reduced
EqMapSmall = np.zeros((map_size/WinMap,map_size/WinMap))
# Contains the whole distributed map
EqMap = np.zeros_like(DistrMap)
# Probability distribution used to weight the cross-entropy during the opt.
P_tilde = EqMapSmall.copy()
# Temperature: 0 -> one-hot vector, 1 -> more mean/saturated
T = 0.01


# Number of discrete regions of LAB colorspace
index = 0
for i in range(0, map_size, WinMap):
	for j in range(0, map_size, WinMap):
		sum = np.sum(DistrMap[i:i + WinMap,j:j + WinMap])
		# If the region has no value whatsoever just ignore it
		if sum > 0:
			# Save the region index in both tables and the corresponding probability
			index += 1
			EqMap[i:i + WinMap,j:j + WinMap] = index
			EqMapSmall[i / WinMap, j / WinMap] = index
			P_tilde[i / WinMap, j / WinMap] = sum
# Number of tiles
Q = index

# Coordinates where the significant bins can be found (C1: array of row indices, C2: array of column indices)
C1, C2 = np.where(EqMapSmall != 0)
# Transform the probability table into a vector of non-zero element
P_tilde = P_tilde[C1, C2]
# Paper formula coefficient (suggested value)
Lambda = 0.5
Weights = 1 / ((1 - Lambda) * P_tilde + (Lambda / Q))
# Normalized such that np.sum(Weights*P_tilde)==1
Weights = Weights / np.sum (P_tilde * Weights)

# Turn back the vector into a matrix (probably better to substitute with a stupid copy)
FinalWeights = np.zeros_like(EqMapSmall)
FinalWeights[C1, C2] = Weights

# Softmax activation function (There is also the default one but is better to have it written for T R A S P A R E N C Y)
def softmax(z):
	return tf.exp(z)/tf.reduce_sum(tf.exp(z))

# Guassian filter definition (There is also the default one but is better to have it written for T R A S P A R E N C Y)
def gaussianFilter(kernlen=21, nsig=3):
	"""Returns a 2D Gaussian kernel array."""

	interval = (2 * nsig + 1.) / (kernlen)
	x = np.linspace(- nsig - interval / 2., nsig + interval / 2., kernlen + 1)
	kern1d = np.diff(stats.norm.cdf(x))
	kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
	kernel = kernel_raw/kernel_raw.sum()
	return kernel

# Function to convert the NxNx2 distr. in a soft-encoding distribution NxNxQ
def lab2distr(image):

	# 2D Gaussian parameters
	kernel = 5
	sigma = 2
	g = gaussianFilter(kernel, sigma)
	res = np.zeros((image.shape[0],image.shape[0],Q))

	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			# create a temporary matrix with added kernel padding to the sides
			tmp = np.zeros((EqMapSmall.shape[0] + (kernel / 2) * 2,
					EqMapSmall.shape[1] + (kernel / 2) * 2))
			# Get the a,b values that will be like coordinates
			# and shift the values to fit the matrix (-127 <= a,b <= 128.
			a, b = image[i,j,:]
			a = int(a) + map_size / 2
		  	b = int(b) + map_size / 2

			colorClass = EqMap[a,b]
			# if non weight is related, error (not 100% sure if needed)
			if colorClass < 1:
				print("This is fucked up!")

			# find the relative center coordinates
			centerX, centerY = np.where(EqMapSmall == colorClass)
			centerX = centerX[0] + kernel/2
			centerY = centerY[0] + kernel/2

			# apply the gaussian filter
			tmp[centerX - kernel / 2: centerX + kernel / 2 + 1,
			    centerY - kernel / 2: centerY + kernel / 2 + 1] = g
			# Remove the borders I added before
			tmp = tmp[kernel / 2: -1 * (kernel / 2), kernel / 2: -1 * (kernel / 2)]
			#return tmp
			# Now I flatten it and take the significant bins.
			res[i,j,:] = tmp[C1, C2]
	return res

# function to convert from NxNxQ to NxNx3 (original image).
def distr2lab(bwimage):

	image = np.zeros((bwimage.shape[0],bwimage.shape[1],2))

	for i in range(bwimage.shape[0]):
		for j in range(bwimage.shape[1]):
			res2 = bwimage[i,j,:]
			# res2=np.exp(np.log(l)/T)/(np.sum(np.exp(np.log(l)/T))) (suspect, reasonable with high probability, investigate)
			matrix = np.zeros_like(EqMapSmall)
			# I put the distribution back to the original colorspace
			matrix[C1,C2] = res2

			c1, c2= np.where(matrix != 0)
			probs = matrix[matrix != 0]
			colorX = np.sum(c1 * (probs * 10)) / np.sum(probs * 10)
			colorY = np.sum(c2 * (probs * 10)) / np.sum(probs * 10)
			
			colorX = colorX * WinMap - map_size / 2
			colorY = colorY * WinMap - map_size / 2

			image[i,j] = [colorX, colorY]

	return image

# function to map weights to final values (kernel excluded)
def mapWeights(batch):
	res = np.zeros((batch.shape[0],batch.shape[1],batch.shape[2]))
	for i in range(batch.shape[0]):
		for j in range(batch.shape[1]):
			for k in range(batch.shape[2]):
				ind1=(int(batch[i,j,k,0]) + 110) / WinMap
				ind2=(int(batch[i,j,k,1]) + 110) / WinMap
				res[i,j,k] = FinalWeights[ind1,ind2]

	return res



def getModel():
	X = tf.placeholder("float",[None, 32,32,1])
	Y = tf.placeholder("float",[None,16,16,Q])
	ZW = tf.placeholder("float",[None,16,16])

	step = int(Q/8)

	with tf.variable_scope("conv1") as scope:
		W = tf.get_variable("W",shape=[3,3,1,step],
			initializer=tf.contrib.layers.xavier_initializer())
		b = tf.get_variable("b",initializer=tf.zeros([step]))
		l = tf.nn.bias_add(
			tf.nn.conv2d(X,W,strides=[1,1,1,1],padding="VALID"),b)
		l_act = tf.nn.relu(l)

	for i in range(2,8):	
		with tf.variable_scope("conv"+str(i)) as scope:
			W = tf.get_variable("W",shape=[3,3,step*(i-1),step*i],
				initializer=tf.contrib.layers.xavier_initializer())
			b = tf.get_variable("b",initializer=tf.zeros([step*i]))
			l = tf.nn.bias_add(
				tf.nn.conv2d(l_act,W,strides=[1,1,1,1],padding="VALID"),b)
			l_act = tf.nn.relu(l)

	with tf.variable_scope("conv8") as scope:
		W = tf.get_variable("W",shape=[3,3,step*7,Q],
			initializer=tf.contrib.layers.xavier_initializer())
		b = tf.get_variable("b",initializer=tf.zeros([Q]))
		output = tf.nn.bias_add(
			tf.nn.conv2d(l,W,strides=[1,1,1,1],padding="VALID"),b)
		#l_act = tf.nn.softmax(l)

	
	pred = tf.nn.softmax(output)
	# I should add the weights
	#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=Y))

	# I guess i should reduce it
	#cost = tf.reduce_mean(ZW*-tf.reduce_sum(Y*tf.log(pred),3)) #16,16,16
	# I get nans
	#When I am getting nans it is usually either of the three:
	# - batch size too small (in your case then just 1)
	# - log(0) somewhere
	#- learning rate too high and uncapped gradients
	cost = tf.reduce_sum(ZW*-tf.reduce_sum(Y*tf.log(pred+eps),3)) #16,16,16
	
        train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

	return (X,Y,ZW),train_step,cost,pred

(X,Y,ZW),train_step,cost,pred=getModel()

saver = tf.train.Saver()

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Training
	epochs = 10
	b_size = 32
	images = []
	for filename in os.listdir("dataset"):
		im = cv2.cvtColor(cv2.imread("dataset/"+filename),cv2.COLOR_BGR2RGB)
		images.append(color.rgb2lab(im))

	iters = [(x,y,z) for z in range(0,256,32) for y in range(0,256,32) for x in range(len(images))]

        for ep in range(epochs):
		random.shuffle(iters)
		cc=0
		print("Epoch: {0}.".format(ep))
		for batch in range(0,len(images),b_size):
			cc+=1
			input_batch = np.zeros((b_size,32,32,1))
			labels_batch = np.zeros((b_size,16,16,Q))
			colors_batch = np.zeros((b_size,16,16,2))
			for index in range(b_size):
				imID,i,j = iters[index]
				piece = images[imID][i:i+32,j:j+32,:]
				input_batch[index,:,:,0] = piece[:,:,0]
				labels_batch[index,:,:,:] = np.reshape(lab2distr(piece[8:-8,8:-8,1:]),(1,16,16,Q))
				colors_batch[index,:,:,:] = np.reshape(piece[8:-8,8:-8,1:],(1,16,16,2))
			#print(colors_batch)
			[_,c]=sess.run([train_step,cost],feed_dict={X:input_batch,Y:labels_batch,ZW:mapWeights(colors_batch)})
			print(c)
				

	saver.save(sess,"test-model-good")		

	# Testing
	for kk in range(10):
		ind = int(random.uniform(0,len(images)))
		res = np.zeros((256,256,3))
		res[:,:,0] = images[ind][:,:,0]
		lala=[];lolo=[]
		for i in range(8,256-8,16):
			for j in range(8,256-8,16):
				inp = np.reshape(images[ind][i-8:i+24,j-8:j+24,0],(1,32,32,1))
				
				[p]=sess.run([pred],feed_dict={X:inp})
				p = np.reshape(p,(16,16,Q))
				lala.append(p)
				lolo.append(inp)
				dp = distr2lab(p)

				res[i:i+16,j:j+16,1:] = dp
				#print(dp.shape)
			

		res_converted = color.lab2rgb(res)
		res2 = np.zeros_like(res)
		res2[:,:,1:] = res[:,:,1:]
		res2[:,:,0] = 100
		res2 = color.lab2rgb(res2)
		plt.figure()
		plt.imshow(res_converted)


--
