import sys
import os
import numpy as np
import torch
import torch.nn as nn

from training_data_wrapper import *


class DrugCellNN(nn.Module):

	def __init__(self, data_wrapper):

		super().__init__()

		self.root = data_wrapper.root
		self.num_hiddens_genotype = data_wrapper.num_hiddens_genotype

		# dictionary from terms to genes directly annotated with the term
		self.term_direct_gene_map = data_wrapper.term_direct_gene_map

		# calculate the number of values in a state (term): term_size_map is the number of all genes annotated with the term
		self.cal_term_dim(data_wrapper.term_size_map)

		# ngenes, gene_dim are the number of all genes
		self.gene_dim = len(data_wrapper.gene_id_mapping)

		# add modules for neural networks to process genotypes
		self.contruct_direct_gene_layer()
		self.construct_NN_graph(data_wrapper.dG)

		# add module for final layer
		self.add_module('final_linear_layer_output', nn.Linear(1, 1))
		#self.add_module('final_linear_layer_output', nn.Linear(data_wrapper.num_hiddens_final, data_wrapper.n_classes))


	# calculate the number of values in a state (term)
	def cal_term_dim(self, term_size_map):

		self.term_dim_map = {}

		for term, term_size in term_size_map.items():
			num_output = self.num_hiddens_genotype

			# log the number of hidden variables per each term
			num_output = int(num_output)
			self.term_dim_map[term] = num_output


	# build a layer for forwarding gene that are directly annotated with the term
	def contruct_direct_gene_layer(self):

		for term, gene_set in self.term_direct_gene_map.items():
			if len(gene_set) == 0:
				print('There are no directed asscoiated genes for', term)
				sys.exit(1)

			# if there are some genes directly annotated with the term, add a layer taking in all genes and forwarding out only those genes
			self.add_module(term+'_direct_gene_layer', nn.Linear(self.gene_dim, len(gene_set)))


	# start from bottom (leaves), and start building a neural network using the given ontology
	# adding modules --- the modules are not connected yet
	def construct_NN_graph(self, dG):

		self.term_layer_list = []   # term_layer_list stores the built neural network
		self.term_neighbor_map = {}

		# term_neighbor_map records all children of each term
		for term in dG.nodes():
			self.term_neighbor_map[term] = []
			for child in dG.neighbors(term):
				self.term_neighbor_map[term].append(child)

		while True:
			leaves = [n for n in dG.nodes() if dG.out_degree(n) == 0]

			if len(leaves) == 0:
				break

			self.term_layer_list.append(leaves)

			for term in leaves:

				# input size will be #chilren + #genes directly annotated by the term
				input_size = 0

				for child in self.term_neighbor_map[term]:
					input_size += self.term_dim_map[child]

				if term in self.term_direct_gene_map:
					input_size += len(self.term_direct_gene_map[term])

				# term_hidden is the number of the hidden variables in each state
				term_hidden = self.term_dim_map[term]

				self.add_module(term+'_linear_layer', nn.Linear(input_size, term_hidden))
				self.add_module(term+'_batchnorm_layer', nn.BatchNorm1d(term_hidden))
				self.add_module(term+'_aux_linear_layer1', nn.Linear(term_hidden,1))
				self.add_module(term+'_aux_linear_layer2', nn.Linear(1,1))

			dG.remove_nodes_from(leaves)


	# definition of forward function
	def forward(self, x):
		gene_input = x.narrow(1, 0, self.gene_dim)

		# define forward function for genotype arm #############################################
		term_gene_out_map = {}

		for term, _ in self.term_direct_gene_map.items():
			term_gene_out_map[term] = self._modules[term + '_direct_gene_layer'](gene_input)

		term_NN_out_map = {}
		aux_out_map = {}

		for i, layer in enumerate(self.term_layer_list):

			for term in layer:

				child_input_list = []

				for child in self.term_neighbor_map[term]:
					child_input_list.append(term_NN_out_map[child])

				if term in self.term_direct_gene_map:
					child_input_list.append(term_gene_out_map[term])

				child_input = torch.cat(child_input_list, 1)

				term_NN_out = self._modules[term+'_linear_layer'](child_input)

				Tanh_out = torch.tanh(term_NN_out)
				term_NN_out_map[term] = self._modules[term+'_batchnorm_layer'](Tanh_out)
				aux_layer1_out = torch.tanh(self._modules[term+'_aux_linear_layer1'](term_NN_out_map[term]))
				aux_out_map[term] = self._modules[term+'_aux_linear_layer2'](aux_layer1_out)

		final_input = term_NN_out_map[self.root]
		aux_out_map['final'] = self._modules['final_linear_layer_output'](final_input)

		return aux_out_map, term_NN_out_map
