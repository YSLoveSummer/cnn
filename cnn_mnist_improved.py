import time

import numpy as np
import random
from tensorflow_core.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# Adam参数
learning_rate = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,\
                0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.001]
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

		# Adam更新参数
		lr = learning_rate[epoch_num]
		# lr = learning_rate[epoch_num] + (learning_rate[epoch_num+1] - learning_rate[epoch_num]) / 430 * t
		self.lr = lr

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
		self.filter_size = [filter_size[2], input_size[2], filter_size[0], filter_size[1]]
		self.filter_weight = np.random.randn(input_size[2]*filter_size[0]*filter_size[1], filter_size[2]) * 0.1
		self.filter_bias = np.ones([1, filter_size[2]]) * 0.1
		self.filter_stride = filter_stride
		self.pad = (filter_size[0] - 1) // 2       # 等宽卷积的pad
		self.pool_size = pool_size[0]         # 池化步长与池化大小一致
		output_height = int(((input_size[0] - filter_size[0] + 2 * self.pad) / filter_stride + 1) // self.pool_size)
		output_width = int(((input_size[1] - filter_size[1] + 2 * self.pad) / filter_stride + 1) // self.pool_size)
		self.output_size = [filter_size[2], output_height, output_width]

		self.V_dw = np.zeros(self.filter_weight.shape)
		self.V_db = np.zeros(self.filter_bias.shape)
		self.S_dw = np.zeros(self.filter_weight.shape)
		self.S_db = np.zeros(self.filter_bias.shape)

	def im2col(self, img, fiter_size, stride=1, pad=0):
		'''
		:param img: 4D array  N,FH,FW,C_{in}
		:param fiter_size: tuple (kh,kw)
		:param stride:
		:return:
		'''
		kh, kw = fiter_size
		self.img_padded = np.pad(img, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
		N, C, H, W = self.img_padded.shape
		out_h = (H - kh) // stride + 1
		out_w = (W - kw) // stride + 1
		col = np.empty((N * out_h * out_w, kw * kh * C))
		outsize = out_w * out_h
		for y in range(out_h):
			y_min = y * stride
			y_max = y_min + kh
			y_start = y * out_w
			for x in range(out_w):
				x_start = x * stride
				x_end = x_start + kw
				col[y_start + x::outsize, :] = self.img_padded[:, :, y_min:y_max, x_start:x_end].reshape(N, -1)
		return col

	def col2im(self, col, input_shape, filter_h, filter_w, stride=1, pad=0):
		"""

		Parameters
		----------
		col :
		input_shape : 输入数据的形状（例：(10, 1, 28, 28)）
		filter_h :
		filter_w
		stride
		pad

		Returns
		-------

		"""
		N, C, H, W = input_shape
		out_h = (H + 2 * pad - filter_h) // stride + 1
		out_w = (W + 2 * pad - filter_w) // stride + 1
		col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

		img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
		for y in range(filter_h):
			y_max = y + stride * out_h
			for x in range(filter_w):
				x_max = x + stride * out_w
				img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

		return img[:, :, pad:H + pad, pad:W + pad]

	def conv2(self, X, filter_weight, filter_bias, stride=1, pad=0):
		'''
		:param X: 4D array  N,FH,FW,C_{in}
		:param W: 4D array  C_{out},kh,kw,C_{in}
		:param stride:
		:param padding:
		:return:   4D array  N,F,H,C_{out}
		'''
		fn, fc, fh, fw = self.filter_size
		N, C, H, W = X.shape
		out_h = (H - fh + 2*pad)// stride + 1
		out_w = (W - fw + 2*pad) // stride + 1
		self.x_col = self.im2col(X, (fh, fw), stride, pad)
		z = np.dot(self.x_col, filter_weight) + filter_bias
		return z.reshape(N, out_h, out_w, fn).transpose(0, 3, 1, 2)

	# 最大池化
	# 采用不重叠池化，池化步长等于池大小
	def max_pool(self, input_array, pool_size):
		(input_num, input_depth, input_height, input_width) = input_array.shape
		output_height = input_height // pool_size
		output_width = input_width // pool_size
		input_col = self.im2col(input_array, (pool_size, pool_size), pool_size).reshape(-1, pool_size*pool_size)
		output = np.max(input_col, axis=1).reshape(input_num, output_height, output_width, input_depth).transpose(0, 3, 1, 2)
		self.input_shape = input_array.shape
		self.argMax = np.argmax(input_col, axis=1)
		self.pool_input_col_shape = input_col.shape
		return output


	# delta_pool是下一层传过来的delta，由此计算传到上一层（卷积层）的delta
	# 池化层不用更新参数
	def pool_backward(self, delta_pool):
		delta_Vector = delta_pool.transpose(0,2,3,1).flatten()
		self.delta_conv = np.zeros(self.pool_input_col_shape)
		self.delta_conv[np.arange(self.argMax.size), self.argMax] = delta_Vector

		self.delta_conv = self.delta_conv.reshape(self.input_shape[0]*self.input_shape[2]*self.input_shape[3], -1)
		self.delta_conv = self.col2im(self.delta_conv, self.input_shape, self.pool_size, self.pool_size, self.pool_size)


	# 计算卷积核权值和偏置的梯度，并更新相关参数
	# 根据池化层传来的delta计算要传到上一层的delta
	def conv_backward(self, t, epoch_num):
		self.delta_conv_col = self.delta_conv.transpose(0, 2, 3, 1).reshape(-1, self.delta_conv.shape[1])

		filter_dw = np.dot(self.x_col.T, self.delta_conv_col) / self.img_num       # im2col后直接矩阵乘法计算卷积核权值梯度
		filter_db = (self.delta_conv.sum(axis=(0,2,3)) / self.img_num).reshape(1, -1)      # 卷积核偏置的平均梯度

		# Adam优化更新卷积核权值和偏置
		lr = learning_rate[epoch_num]

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

		# 卷积核翻转180度
		# flip_filter = np.zeros(self.filter_weight.shape)
		# for i in range(flip_filter.shape[0]):
		# 	for j in range(flip_filter.shape[1]):
		# 		flip_filter[i,j,:,:] = np.fliplr(np.flipud(self.filter_weight[i,j,:,:]))


		################### 只有一层卷积层时可不使用
		# 计算传给上一层的delta，就是input的delta
		# shape = self.filter_weight.shape
		# filter_converse = np.zeros((shape[1], shape[0], shape[2], shape[3]))    # 先将filter前两个维度倒换一下，以便使用conv函数
		# for i in range(shape[1]):
		# 	filter_converse[i,:,:,:] = self.filter_weight[:,i,:,:]
		# bias = np.zeros((filter_depth, 1))
		# filter和这一层delta卷积得到上一层delta
		# self.delta_input = self.conv(self.delta_conv, filter_converse, bias, self.filter_stride, self.pad)


	# 向前传播，计算卷积并池化后的输出
	def forward(self, input_array):
		self.img_num = input_array.shape[0]
		self.conv_output = self.conv2(input_array, self.filter_weight, self.filter_bias, self.filter_stride, self.pad)
		self.conv_output_activate = Relu(self.conv_output)        # 卷积后激活
		pool_output = self.max_pool(self.conv_output_activate, self.pool_size)   # 最大池化
		# self.output_array = pool_output
		return pool_output

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
		batch_nums = len(X) // batch_size
		for i in range(epoch_num):
			begin = time.time()
			sample = random.sample(range(len(X)), len(X))
			x_epoch = X[sample]
			y_epoch = y[sample]
			for j in range(batch_nums):
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
					print('%2d%% batch %4d, batch loss=%.3f' % (j*100//batch_nums+1, j+1 + i*batch_nums, loss), end='  ')
					print('batch accuracy = %.3f' % accuracy, end='  ')
					print('lr = %.4f' % self.fclayer.lr)
			end = time.time()
			print('———————————— epoch %d, batch loss = %.3f' % (i+1, loss), end='  ')
			print('batch accuracy = %.3f,  run time = %.3fs ————————————' % (accuracy, end-begin))

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
	filter_size = [3, 3, 20]
	filter_stride = 1
	pool_size = [2,2]
	FC_hidden_size = 512
	output_size = 10
	cnn = CNN(input_size, filter_size, filter_stride, pool_size, FC_hidden_size, output_size)

	epoch_num = 4
	batch_size = 30
	loss_list, accuracy_list = cnn.fit(train_data, train_labels, epoch_num, batch_size)

	_, y_prob = cnn.predict(test_data)
	accuracy = np.mean(np.argmax(y_prob, axis=1) == np.argmax(test_labels, axis=1))
	print('测试集 accuracy = %.3f' % accuracy)
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
