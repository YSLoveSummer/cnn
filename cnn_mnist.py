import time

import numpy as np
import random
from tensorflow_core.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# Adam参数
learning_rate = [0.01, 0.01, 0.005, 0.005, 0.0001, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005]
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8


# relu激活函数
def Relu(x):
	z = x
	z[z<0] = 0
	return z

def dRelu(x):
	dx = x
	dx[dx<=0] = 0
	dx[dx>0] = 1
	return dx

def softfmax(z):
	y = np.exp(z)
	S = np.sum(y, axis=1, keepdims=True)
	return y/S

# 交叉熵损失
def cross_entropy(y_prob, y):
	z = y * np.log(y_prob)
	loss = np.mean(-np.sum(z, axis=1))
	return loss

# 交叉熵对softmax净输入的导数（链式求导的结果）
def delta_softmax(y_prob, y):
	return y_prob - y

# 对标签y进行One-hot编码
def one_hot(y):
	one_hot = np.zeros((len(y), len(np.unique(y))))
	one_hot[np.arange(len(y)), y] = 1
	return one_hot

# 对标签y进行one_hot解码
def de_one_hot(y):
	de_one_hot = np.where(y==1)[1]
	return de_one_hot


# 全连接层
class full_connect_layer:
	def __init__(self, input_size, hidden_size, output_size):
		self.weight_input = np.random.randn(input_size, hidden_size) * 0.1
		self.bias_input = np.ones((1, hidden_size)) * 0.1
		self.weight_hidden = np.random.randn(hidden_size, output_size) * 0.1
		self.bias_hidden = np.ones((1, output_size)) * 0.1
		self.loss_list = []
		self.V_dw_hidden = np.zeros(self.weight_hidden.shape)
		self.V_dw_input = np.zeros(self.weight_input.shape)
		self.V_db_hidden = np.zeros(self.bias_hidden.shape)
		self.V_db_input = np.zeros(self.bias_input.shape)
		self.S_dw_hidden = np.zeros(self.weight_hidden.shape)
		self.S_dw_input = np.zeros(self.weight_input.shape)
		self.S_db_hidden = np.zeros(self.bias_hidden.shape)
		self.S_db_input = np.zeros(self.bias_input.shape)

	def forward_backward(self, X, y, t, epoch_num):
		z_input = np.dot(X, self.weight_input) + self.bias_input    # 隐层净输入
		x_hidden = Relu(z_input)            # 激活后神经元节点值
		z_hidden = np.dot(x_hidden, self.weight_hidden) + self.bias_hidden  # 输出层净输入
		y_prob = softfmax(z_hidden)

		loss = cross_entropy(y_prob, y)
		self.loss_list.append(loss)

		# 计算每层误差项
		delta_output = delta_softmax(y_prob, y)
		delta_hidden = np.multiply(dRelu(z_input), np.dot(delta_output, self.weight_hidden.T))
		self.delta_input = np.dot(delta_hidden, self.weight_input.T)           ### 要传给池化层的delta

		# 权值更新
		# dw_hidden = np.dot(x_hidden.T, delta_output) / len(y)      # 权值平均梯度(这里注意要除以输入样本的个数）
		# db_hidden = np.sum(delta_output, axis=0) / len(y)        # 偏置平均梯度
		# self.weight_hidden -= learning_rate * dw_hidden
		# self.bias_hidden -= learning_rate * db_hidden
		# dw_input = np.dot(X.T, delta_hidden) / len(y)
		# db_input = np.sum(delta_hidden, axis=0) / len(y)
		# self.weight_input -= learning_rate * dw_input
		# self.bias_input -= learning_rate * db_input

		# Adam更新参数
		lr = learning_rate[epoch_num]
		# lr = learning_rate[epoch_num] + (learning_rate[epoch_num+1] - learning_rate[epoch_num]) / 430 * t

		dw_hidden = np.dot(x_hidden.T, delta_output) / len(y)      # 权值平均梯度(这里注意要除以输入样本的个数）
		db_hidden = np.sum(delta_output, axis=0) / len(y)        # 偏置平均梯度
		dw_input = np.dot(X.T, delta_hidden) / len(y)
		db_input = np.sum(delta_hidden, axis=0) / len(y)

		self.V_dw_hidden = beta1 * self.V_dw_hidden + (1 - beta1) * dw_hidden
		self.V_dw_input = beta1 * self.V_dw_input + (1 - beta1) * dw_input
		self.V_db_hidden = beta1 * self.V_db_hidden + (1 - beta1) * db_hidden
		self.V_db_input = beta1 * self.V_db_input + (1 - beta1) * db_input

		V_dw_hidden_correct = self.V_dw_hidden / (1 - np.power(beta1, t))
		V_dw_input_correct = self.V_dw_input / (1 - np.power(beta1, t))
		V_db_hidden_correct = self.V_db_hidden / (1 - np.power(beta1, t))
		V_db_input_correct = self.V_db_input / (1 - np.power(beta1, t))

		self.S_dw_hidden = beta2 * self.S_dw_hidden + (1 - beta2) * np.power(dw_hidden, 2)
		self.S_dw_input = beta2 * self.S_dw_input + (1 - beta2) * np.power(dw_input, 2)
		self.S_db_hidden = beta2 * self.S_db_hidden + (1 - beta2) * np.power(db_hidden, 2)
		self.S_db_input = beta2 * self.S_db_input + (1 - beta2) * np.power(db_input, 2)

		S_dw_hidden_correct = self.S_dw_hidden / (1 - np.power(beta2, t))
		S_dw_input_correct = self.S_dw_input / (1 - np.power(beta2, t))
		S_db_hidden_correct = self.S_db_hidden / (1 - np.power(beta2, t))
		S_db_input_correct = self.S_db_input / (1 - np.power(beta2, t))

		self.weight_hidden -= lr * V_dw_hidden_correct / np.sqrt(S_dw_hidden_correct + epsilon)
		self.bias_hidden -= lr * V_db_hidden_correct / np.sqrt(S_db_hidden_correct + epsilon)

		self.weight_input -= lr * V_dw_input_correct / np.sqrt(S_dw_input_correct + epsilon)
		self.bias_input -= lr * V_db_input_correct / np.sqrt(S_db_input_correct + epsilon)

		return self.delta_input, loss, y_prob

	def predict(self, X):
		z_input = np.dot(X, self.weight_input) + self.bias_input
		x_hidden = Relu(z_input)
		z_hidden = np.dot(x_hidden, self.weight_hidden) + self.bias_hidden
		y_prob = softfmax(z_hidden)

		y_pred = np.zeros(y_prob.shape)
		index = np.argmax(y_prob, axis=1)
		y_pred[np.arange(len(y_pred)), index] = 1

		return y_pred, y_prob


class ConvLayer:
	# 一层卷积，一层池化
	def __init__(self, input_size, filter_size, filter_stride, pool_size):
		self.filter_weight = np.random.randn(filter_size[2], input_size[2], filter_size[0], filter_size[1]) * 0.1
		self.filter_bias = np.ones([filter_size[2], 1]) * 0.1
		self.filter_stride = filter_stride
		self.padding = (filter_size[0] - 1) // 2       # 等宽卷积的padding
		self.pool_size = pool_size[0]         # 池化步长与池化大小一致
		output_height = int(((input_size[0] - filter_size[0] + 2 * self.padding) / filter_stride + 1) // self.pool_size)
		output_width = int(((input_size[1] - filter_size[1] + 2 * self.padding) / filter_stride + 1) // self.pool_size)
		self.output_size = [filter_size[2], output_height, output_width]

		self.V_dw = np.zeros(self.filter_weight.shape)
		self.V_db = np.zeros(self.filter_bias.shape)
		self.S_dw = np.zeros(self.filter_weight.shape)
		self.S_db = np.zeros(self.filter_bias.shape)


	def zero_padding(self, input_array, padding):
		(input_num, input_depth, input_height, input_width) = input_array.shape
		padded_array = np.zeros((input_num, input_depth, input_height + 2*padding, input_width + 2*padding))
		padded_array[:, :, padding: padding + input_height, padding: padding + input_width] = input_array
		return padded_array

	def conv(self, input_array, filter, bias, stride, padding):
		(input_num, input_depth, input_height, input_width) = input_array.shape
		(filter_num, filter_depth, filter_height, filter_width) = filter.shape
		output_height = int((input_height - filter_height + 2 * padding) / stride + 1)
		output_width = int((input_width - filter_width + 2 * padding) / stride + 1)
		output_array = np.zeros((input_num, filter_num, output_height, output_width))
		padded_input = self.zero_padding(input_array, padding)
		self.padded_input = padded_input
		# 遍历卷积区域
		for h in range(output_height):
			for w in range(output_width):
				for d in range(filter_num):
					input_slice = padded_input[:, :, h*stride : h*stride+filter_height, w*stride : w*stride+filter_width]
					output_array[:, d, h, w] = self.conv_caculate(input_slice, filter[d], bias[d])
		return output_array


	def conv_caculate(self, input_slice, filter, bias):
		z = np.multiply(input_slice, filter)
		s = z.sum(axis=(1,2,3)) + [bias]
		return s

	# 最大池化
	# 采用不重叠池化，池化步长等于池大小
	def max_pool(self, input_array, pool_size):
		(input_num, input_depth, input_height, input_width) = input_array.shape
		output_height = input_height // pool_size
		output_width = input_width // pool_size
		output_array = np.zeros((input_num, input_depth, output_height, output_width))
		for h in range(output_height):
			for w in range(output_width):
				input_slice = input_array[:, :, h*pool_size : h*pool_size+pool_size, w*pool_size : w*pool_size+pool_size]
				output_array[:, :, h, w] = input_slice.max(axis=(2,3))
		return output_array

	# 向前传播，计算卷积并池化后的输出
	def forward(self, input_array):
		self.input_array = input_array
		self.conv_output = self.conv(input_array, self.filter_weight, self.filter_bias, self.filter_stride, self.padding)
		self.conv_output_activate = Relu(self.conv_output)        # 卷积后激活
		pool_output = self.max_pool(self.conv_output_activate, self.pool_size)   # 最大池化
		self.output_array = pool_output
		return pool_output

	# delta_pool是下一层传过来的delta，由此计算传到上一层（卷积层）的delta
	# 池化层不用更新参数
	def pool_backward(self, delta_pool):
		(delta_num, delta_depth, delta_height, delta_width) = delta_pool.shape
		self.delta_conv = np.zeros(self.conv_output_activate.shape)
		for h in range(delta_height):
			for w in range(delta_width):
				conv_slice = self.conv_output_activate[:, :, h: h+self.pool_size, w: w+self.pool_size]
				Max_pos = (conv_slice==np.max(conv_slice))
				self.delta_conv[:, :, h: h+self.pool_size, w: w+self.pool_size] = Max_pos * delta_pool[:, :, h:h+1, w:w+1]
		self.delta_conv = np.multiply(dRelu(self.conv_output), self.delta_conv)    # 乘以激活函数对激活前净输入的导数，这个就是要传到上一层卷积层的delta

	# 计算卷积核权值和偏置的梯度，并更新相关参数
	# 根据池化层传来的delta计算要传到上一层的delta
	def conv_backward(self, t, epoch_num):
		(delta_num, delta_depth, delta_height, delta_width) = self.delta_conv.shape
		(filter_num, filter_depth, filter_height, filter_width) = self.filter_weight.shape
		filter_dw = np.zeros(self.filter_weight.shape)
		padded_input = self.padded_input
		# 卷积核翻转180度
		# flip_filter = np.zeros(self.filter_weight.shape)
		# for i in range(flip_filter.shape[0]):
		# 	for j in range(flip_filter.shape[1]):
		# 		flip_filter[i,j,:,:] = np.fliplr(np.flipud(self.filter_weight[i,j,:,:]))
		# 遍历卷积区域
		# delta和X做卷积，这里步长只考虑等于1的情形（不具有普遍性）
		for n in range(filter_num):
			for h in range(filter_height):
				for w in range(filter_width):
					input_slice = padded_input[:, :, h:h+delta_height, w:w+delta_width]
					# 卷积核权值的平均梯度（除以输入样本个数）
					filter_dw[n, :, h, w] = np.multiply(input_slice, self.delta_conv[:, n:n+1, :, :]).sum(axis=(0,2,3)) / self.input_array.shape[0]
		filter_db = (self.delta_conv.sum(axis=(0,2,3)) / self.input_array.shape[0]).reshape(-1, 1)      # 卷积核偏置的平均梯度
		# 更新卷积核权值和偏置
		# self.filter_weight -= learning_rate * filter_dw
		# self.filter_bias -= learning_rate * filter_db

		# Adam优化更新卷积核权值和偏置
		lr = learning_rate[epoch_num]
		# lr = learning_rate[epoch_num] + (learning_rate[epoch_num+1] - learning_rate[epoch_num]) / 430 * t

		self.V_dw= beta1 * self.V_dw + (1 - beta1) * filter_dw
		self.V_db = beta1 * self.V_db + (1 - beta1) * filter_db

		V_dw_correct = self.V_dw / (1 - np.power(beta1, t))
		V_db_correct = self.V_db / (1 - np.power(beta1, t))

		self.S_dw = beta2 * self.S_dw + (1 - beta2) * np.power(filter_dw, 2)
		self.S_db = beta2 * self.S_db + (1 - beta2) * np.power(filter_db, 2)

		S_dw_correct = self.S_dw / (1 - np.power(beta2, t))
		S_db_correct = self.S_db / (1 - np.power(beta2, t))

		self.filter_weight -= lr * V_dw_correct / np.sqrt(S_dw_correct + epsilon)
		self.filter_bias -= lr * V_db_correct / np.sqrt(S_db_correct + epsilon)


		################### 只有一层卷积层时可不使用
		# 计算传给上一层的delta，就是input的delta
		# shape = self.filter_weight.shape
		# filter_converse = np.zeros((shape[1], shape[0], shape[2], shape[3]))    # 先将filter前两个维度倒换一下，以便使用conv函数
		# for i in range(shape[1]):
		# 	filter_converse[i,:,:,:] = self.filter_weight[:,i,:,:]
		# bias = np.zeros((filter_depth, 1))
		# filter和这一层delta卷积得到上一层delta
		# self.delta_input = self.conv(self.delta_conv, filter_converse, bias, self.filter_stride, self.padding)


	def backward(self, delta_pool, t, epoch_num):
		self.pool_backward(delta_pool)
		self.conv_backward(t, epoch_num)
		# return self.delta_input



class CNN:
	def __init__(self, input_size, filter_size, filter_stride, pool_size, FC_hidden_size, FC_output_size):
		self.convlayer = ConvLayer(input_size, filter_size, filter_stride, pool_size)
		FC_input_size = int(np.prod(self.convlayer.output_size))      # 全连接输入层节点个数
		self.fclayer = full_connect_layer(FC_input_size, FC_hidden_size, FC_output_size)

	def fit(self, X, y, epoch_num, batch_size=30):
		self.convlayer.output_size.insert(0, batch_size)
		accuracy_list = []
		for i in range(epoch_num):
			begin = time.time()
			sample = random.sample(range(len(X)), len(X))
			x_epoch = X[sample]
			y_epoch = y[sample]
			for j in range(len(X) // batch_size):
				x_batch = x_epoch[batch_size * j: batch_size * (j+1)]
				y_batch = y_epoch[batch_size * j: batch_size * (j+1)]
				# 前向传播
				conv_output = self.convlayer.forward(x_batch)
				FC_input = conv_output.reshape(conv_output.shape[0], -1)
				delta_FC, loss, y_prob = self.fclayer.forward_backward(FC_input, y_batch, j+1, i)
				# 误差逆传播，并更新参数
				delta_pool = delta_FC.reshape(self.convlayer.output_size)
				self.convlayer.backward(delta_pool, j+1, i)
				accuracy = np.mean(np.argmax(y_prob, axis=1) == np.argmax(y_batch, axis=1))
				accuracy_list.append(accuracy)
				if (j%20==0):
					print('batch %d, batch loss=%.4f' % (j+1 + i*(len(X) // batch_size), loss), end=' ')
					print('batch accuracy = %.4f' % accuracy, end=' ')
					print('lr = %f' % self.fclayer.lr)
			end = time.time()
			print('———————————— epoch %d, batch loss = %.4f' % (i+1, loss), end=' ')
			print('batch accuracy = %.4f,  run time = %.4fs ————————————' % (accuracy, end-begin))

		return self.fclayer.loss_list, accuracy_list


	def predict(self, X):
		conv_output = self.convlayer.forward(X)
		FC_input = conv_output.reshape(conv_output.shape[0], -1)
		return self.fclayer.predict(FC_input)



if __name__ == '__main__':
	mnist = input_data.read_data_sets("MNIST_data", one_hot=True, validation_size=0)
	# 训练集和测试集
	train_data = mnist.train.images
	train_labels = mnist.train.labels
	test_data = mnist.test.images
	test_labels = mnist.test.labels

	train_data = train_data.reshape(train_data.shape[0], 1, 28, 28)
	test_data = test_data.reshape(test_data.shape[0], 1, 28, 28)

	input_size = [28, 28, 1]
	filter_size = [3, 3, 10]
	filter_stride = 1
	pool_size = [2,2]
	FC_hidden_size = 100
	output_size = 10
	cnn = CNN(input_size, filter_size, filter_stride, pool_size, FC_hidden_size, output_size)

	epoch_num = 4
	batch_size = 128
	loss_list, accuracy_list = cnn.fit(train_data, train_labels, epoch_num, batch_size)

	_, y_prob = cnn.predict(test_data)
	accuracy = np.mean(np.argmax(y_prob, axis=1) == np.argmax(test_labels, axis=1))
	print('测试集 accuracy = %.4f' % accuracy)
	plt.plot(loss_list)
	plt.xlabel('batch_nums')
	plt.ylabel('batch_loss')
	plt.title('batch_size=%d' %(batch_size))
	plt.show()

	plt.plot(accuracy_list)
	plt.xlabel('batch_nums')
	plt.ylabel('batch_accuracy')
	plt.title('batch_size=%d' %(batch_size))
	plt.show()
