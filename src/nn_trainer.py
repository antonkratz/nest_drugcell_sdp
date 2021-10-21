import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as du
from torch.autograd import Variable

import util
from training_data_wrapper import *
from drugcell_nn import *
from predict_drugcell import *


class NNTrainer():

	def __init__(self, opt):
		self.data_wrapper = TrainingDataWrapper(opt)
		self.model = DrugCellNN(self.data_wrapper)
		self.model.cuda(self.data_wrapper.cuda)


	def train_model(self):

		epoch_start_time = time.time()
		max_corr = 0

		train_feature, train_label, val_feature, val_label = self.data_wrapper.prepare_train_data()

		term_mask_map = util.create_term_mask(self.model.term_direct_gene_map, self.model.gene_dim, self.data_wrapper.cuda)
		for name, param in self.model.named_parameters():
			term_name = name.split('_')[0]
			if '_direct_gene_layer.weight' in name:
				param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
			else:
				param.data = param.data * 0.1

		train_label_gpu = Variable(train_label.cuda(self.data_wrapper.cuda))
		val_label_gpu = Variable(val_label.cuda(self.data_wrapper.cuda))
		train_loader = du.DataLoader(du.TensorDataset(train_feature, train_label), batch_size=self.data_wrapper.batchsize, shuffle=True)

		optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.data_wrapper.lr, betas=(0.9, 0.99), eps=1e-05, weight_decay=self.data_wrapper.wd)
		optimizer.zero_grad()

		for epoch in range(self.data_wrapper.epochs):
			# Train
			self.model.train()
			train_predict = torch.zeros(0, 0).cuda(self.data_wrapper.cuda)

			for i, (inputdata, labels) in enumerate(train_loader):
				#features = util.build_input_vector(inputdata, self.data_wrapper.cell_features, self.data_wrapper.drug_features)
				features = util.build_input_vector(inputdata, self.data_wrapper.cell_features)
				cuda_features = Variable(features.cuda(self.data_wrapper.cuda))
				cuda_labels = Variable(labels.cuda(self.data_wrapper.cuda))

				# Forward + Backward + Optimize
				optimizer.zero_grad()  # zero the gradient buffer

				aux_out_map,_ = self.model(cuda_features)

				if train_predict.size()[0] == 0:
					train_predict = aux_out_map['final'].data
				else:
					train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)

				total_loss = 0
				for name, output in aux_out_map.items():
					loss = nn.MSELoss()
					if name == 'final':
						total_loss += loss(output, cuda_labels)
					else:
						total_loss += self.data_wrapper.alpha * loss(output, cuda_labels)
				total_loss.backward()

				for name, param in self.model.named_parameters():
					if '_direct_gene_layer.weight' not in name:
						continue
					term_name = name.split('_')[0]

				optimizer.step()

			train_corr = util.pearson_corr(train_predict, train_label_gpu)
			#train_corr = util.get_drug_corr_median(train_predict, train_label_gpu, train_feature)

			epoch_end_time = time.time()
			print("epoch {}\ttrain_corr {:.5f}\ttotal_loss {:.3f}\telapsed_time {}".format(epoch, train_corr, total_loss, epoch_end_time - epoch_start_time))
			epoch_start_time = epoch_end_time

		return self.predict_cell(val_feature)


	def predict_cell(self, predict_feature):

		self.model.eval()

		features = util.build_input_vector(predict_feature, self.data_wrapper.cell_features)
		cuda_features = Variable(features.cuda(self.data_wrapper.cuda), requires_grad=False)

		# make prediction for test data
		aux_out_map,_ = model(cuda_features)
		test_predict = aux_out_map['final'].data
		return test_predict.cpu().numpy()
