import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm

from load_graph import load_reddit, inductive_split, load_ogb
from memory_usage import see_memory_usage
import tracemalloc
from cpu_mem_usage import get_memory


def ttt(tic, str1):
	toc = time.time()
	print(str1 + ' step Time(s): {:.4f}'.format(toc - tic))
	return toc


# def CPU_mem(str1):
# 	current, peak = tracemalloc.get_traced_memory()
# 	print(f"{str1} Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
# 	snapshot = tracemalloc.take_snapshot()
# 	top_stats = snapshot.statistics('lineno')

# 	print("[ Top 10 ]")
# 	for stat in top_stats[:10]:
# 		print(stat)
# 	tracemalloc.stop()
# 	tracemalloc.start()

# 	return


# def load_data(dset):
#     num_v = 0
#     num_e = 0
#     with open("/home/cc/GNN-Computing-master/data/{}.config".format(dset), 'r') as f:
#         l = f.readline().split(' ')
#         num_v = (int)(l[0])
#         num_e = (int)(l[1])

#     src_list = []
#     dst_list = []
#     t_load_begin = time.time()
#     with open("./home/cc/GNN-Computing-master/data/{}.graph".format(dset), 'r') as f:
#         ptr = f.readline().strip("\n").strip(" ").split(" ")
#         idx = f.readline().strip("\n").strip(" ").split(" ")
#         for item in range(num_v):
#             which = (int)(item)
#             selfloop = False
#             for i in range((int)(ptr[which]), (int)(ptr[which + 1])):
#                 dst_list.append(which)
#                 src_list.append((int)(idx[i]))
#                 if which == (int)(idx[i]):
#                     selfloop = True
#     g = DGLGraph((src_list, dst_list)).to(device)
#     return g

# aggre = 'lstm'
aggre = 'mean'


class SAGE(nn.Module):
	def __init__(self,
	             in_feats,
	             n_hidden,
	             n_classes,
	             n_layers,
	             activation,
	             dropout):
		super().__init__()
		self.n_layers = n_layers
		self.n_hidden = n_hidden
		self.n_classes = n_classes
		self.layers = nn.ModuleList()
		self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, aggre))
		for i in range(1, n_layers - 1):
			self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, aggre))
		self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, aggre))
		self.dropout = nn.Dropout(dropout)
		self.activation = activation

	def forward(self, blocks, x):
		h = x
		for l, (layer, block) in enumerate(zip(self.layers, blocks)):
			h = layer(block, h)
			if l!=len(self.layers) - 1:
				h = self.activation(h)
				h = self.dropout(h)
		return h

	def inference(self, g, x, device):
		"""
		Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
		g : the entire graph.
		x : the input of entire node set.

		The inference code is written in a fashion that it could handle any number of nodes and
		layers.
		"""
		# During inference with sampling, multi-layer blocks are very inefficient because
		# lots of computations in the first few layers are repeated.
		# Therefore, we compute the representation of all nodes layer by layer.  The nodes
		# on each layer are of course splitted in batches.
		# TODO: can we standardize this?
		for l, layer in enumerate(self.layers):
			y = th.zeros(g.num_nodes(), self.n_hidden if l!=len(self.layers) - 1 else self.n_classes)

			sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
			dataloader = dgl.dataloading.NodeDataLoader(
				g,
				th.arange(g.num_nodes()),
				sampler,
				batch_size=args.batch_size,
				shuffle=True,
				drop_last=False,
				num_workers=args.num_workers)

			for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
				block = blocks[0]

				block = block.int().to(device)
				h = x[input_nodes].to(device)
				h = layer(block, h)
				if l!=len(self.layers) - 1:
					h = self.activation(h)
					h = self.dropout(h)

				y[output_nodes] = h.cpu()

			x = y
		return y


def compute_acc(pred, labels):
	"""
	Compute the accuracy of prediction given the labels.
	"""
	labels = labels.long()
	return (th.argmax(pred, dim=1)==labels).float().sum() / len(pred)


def evaluate(model, g, nfeat, labels, val_nid, device):
	"""
	Evaluate the model on the validation set specified by ``val_nid``.
	g : The entire graph.
	inputs : The features of all the nodes.
	labels : The labels of all the nodes.
	val_nid : the node Ids for validation.
	device : The GPU device to evaluate on.
	"""
	model.eval()
	with th.no_grad():
		pred = model.inference(g, nfeat, device)
	model.train()
	return compute_acc(pred[val_nid], labels[val_nid].to(pred.device))


def load_subtensor(nfeat, labels, seeds, input_nodes, device):
	"""
	Extracts features and labels for a subset of nodes
	"""
	batch_inputs = nfeat[input_nodes].to(device)
	batch_labels = labels[seeds].to(device)
	return batch_inputs, batch_labels


#### Entry point
def run(args, device, data, tic):
	# Unpack data
	# get_memory("-----------------------------------------start run")


	t_1 = ttt(tic, " start_run")
	n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
	val_nfeat, val_labels, test_nfeat, test_labels = data
	t_2 = ttt(t_1, " unpack data")
	# get_memory("-----------------------------------------after unpack data")

	in_feats = train_nfeat.shape[1]
	train_nid = th.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
	t_3 = ttt(t_2, " train_mask")
	val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
	test_nid = th.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]
	t_4 = ttt(t_3, "val_mask and  test_mask")
	# get_memory("-----------------------------------------after add mask")

	print("in_feats " + str(in_feats))
	# print("train_g.shape "+ str(train_g.shape))
	print("train_labels.shape " + str(train_labels.shape))
	# print("val_g.shape "+ str(val_g.shape))

	# Create PyTorch DataLoader for constructing blocks
	# get_memory("-----------------------------------------before sampler (MB)")
	sampler = dgl.dataloading.MultiLayerNeighborSampler(
		[int(fanout) for fanout in args.fan_out.split(',')])
	t_5 = ttt(t_4, " create a sampler instance")
	# see_memory_usage("-----------------------------------------after sampler------------------------")
	# get_memory("-----------------------------------------before load_data (MB)")
	dataloader = dgl.dataloading.NodeDataLoader(
		train_g,
		train_nid,
		sampler,
		batch_size=args.batch_size,
		shuffle=True,
		drop_last=False,
		num_workers=args.num_workers)
	print("args.batch_size " + str(args.batch_size))
	t_6 = ttt(t_5, " create a data loader instance")
	# get_memory("-----------------------------------------after dataloader (MB)")
	# see_memory_usage("-----------------------------------------after data loader------------------------")

	# Define model and optimizer
	model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
	t_7 = ttt(t_6, " create a model instance")
	print(model)
	# see_memory_usage("-----------------------------------------before model to gpu------------------------")
	model = model.to(device)
	loss_fcn = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	# see_memory_usage("-----------------------------------------before start------------------------")
	t_8 = ttt(t_7, " create an optimizer instance")

	# Training loop
	avg = 0
	iter_tput = []
	time1_list = []
	avg_step_time = []
	# tracemalloc.start()

	# CPU_mem("-----------------------------------------before start------------------------")
	for epoch in range(args.num_epochs):
		tic = time.time()
		time3_list = []

		# Loop over the dataloader to sample the computation dependency graph as a list of
		# blocks.
		# th.cuda.synchronize()
		#
		# start = th.cuda.Event(enable_timing=True)
		# end = th.cuda.Event(enable_timing=True)
		# start.record()
		tic_step = time.time()
		# get_memory("-----------------------------------------before for loop ")
		th.cuda.synchronize()
		for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
			print("\n   ***************************     step   " + str(step) + "   *************************************" )
			# get_memory("-----------------------------------------after start a new step")
			th.cuda.synchronize()
			start = th.cuda.Event(enable_timing=True)
			end = th.cuda.Event(enable_timing=True)
			start.record()

			# if step>1:
			#     break

			# print("*"*80 +str(step))
			# print("input_nodes.shape "+str(input_nodes.shape))
			# print("output_nodes.shape "+str(seeds.shape))
			# print("blocks.length "+str(len(blocks)))
			# print("blocks.shape "+str(blocks.shape))
			# print(blocks)

			# see_memory_usage("-----------------------------------------step start------------------------")
			# Load the input features as well as output labels
			# t1 = ttt(tic_step, "after start new step-----***************************-------------------")
			batch_inputs, batch_labels = load_subtensor(train_nfeat, train_labels, seeds, input_nodes, device)
			# t2 = ttt(t1, "after load_subtensor")
			# get_memory("-----------------------------------------after load_subtensor (MB)")
			# CPU_mem("-----------------------------------------before blocks to device")
			# see_memory_usage("-----------------------------------------before blocks to device")
			blocks = [block.int().to(device) for block in blocks]

			# t3 = ttt(t2, "after block")
			get_memory("-----------------------------------------after blocks (MB)")
			# CPU_mem("-----------------------------------------after blocks to device")
			# see_memory_usage("-----------------------------------------after blocks to device")
			# print("---------------------------------------------------------batch_inputs.shape " + str(
			# batch_inputs.shape))
			th.cuda.synchronize()  # wait for move to complete

			end.record()
			th.cuda.synchronize()

			time3_list.append(start.elapsed_time(end))
			# start = th.cuda.Event(enable_timing=True)
			# end = th.cuda.Event(enable_timing=True)
			# start.record()

			# Compute loss and prediction
			batch_pred = model(blocks, batch_inputs)
			# t4 = ttt(t3, "after batch train")
			get_memory("-----------------------------------------after batch train (MB)")

			# CPU_mem("-----------------------------------------after batch train")
			# see_memory_usage("-----------------------------------------after batch train")
			loss = loss_fcn(batch_pred, batch_labels)
			# t5 = ttt(t4, "after loss function")
			get_memory("-----------------------------------------after loss function (MB)")
			# see_memory_usage("-----------------------------------------after batch loss")
			# CPU_mem("-----------------------------------------after batch loss")
			optimizer.zero_grad()
			# t6 = ttt(t5, "after zero_grad")
			get_memory("-----------------------------------------after optimizer.zero_grad (MB)")

			loss.backward()
			# t7 = ttt(t6, "after backward")
			get_memory("-----------------------------------------after batch loss backward (MB)")
			# see_memory_usage("-----------------------------------------after batch loss backward")
			# CPU_mem("-----------------------------------------after batch loss backward")
			optimizer.step()
			# t8 = ttt(t7, "after optimizer step")
			get_memory("-----------------------------------------after optimizer.step (MB)")
			tic_step = time.time()
			th.cuda.synchronize()  # wait for all training steps to complete
			# end.record()
			# th.cuda.synchronize()
			#
			# time1_list.append(start.elapsed_time(end))
			# time2_list.append(start.elapsed_time(end))
			step_time = time.time() - tic_step
			avg_step_time.append(step_time)
			# print(step_time)

			iter_tput.append(len(seeds) / (time.time() - tic_step))
			tic_step = time.time()
		print('avg iteration data from cpu to GPU time:%.8f' % (sum(time3_list[5:]) / len(time3_list[5:])))
		th.cuda.synchronize()
		time1_list.append(sum(time3_list[5:]) / len(time3_list[5:]))
		# print('avg iteration GPU training time:%.8f' % (sum(time2_list[5:]) / len(time2_list[5:])))
		# th.cuda.synchronize()  # wait for all training steps to complete
		# end.record()
		# th.cuda.synchronize()
		# print('current epoch GPU training time:%.8f' % start.elapsed_time(end))
		# time1_list.append(start.elapsed_time(end))

		# iter_tput.append(len(seeds) / (time.time() - tic_step))
		# if step % args.log_every==0:
		# 	print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
		# 			epoch, step, loss.item(), 0, np.mean(iter_tput[3:]), 0))
		# 	# acc = compute_acc(batch_pred, batch_labels)
		# 	# gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
		# 	# print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
		# 	#     epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
		# 	# print(a
		# 	# 	'Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
		# 	# 		epoch, step, loss.item(), 0, np.mean(iter_tput[3:]), gpu_mem_alloc))
		#     # print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
		# 	# 		epoch, step, loss.item(), 0, np.mean(iter_tput[3:]), 0))
		# tic_step = time.time()

		toc = time.time()
		print('Epoch Time(s): {:.4f}'.format(toc - tic))
		# avg += toc - tic
		if epoch >= 5:
			avg += toc - tic
	# if epoch % args.eval_every==0 and epoch!=0:
	# 	eval_acc = evaluate(model, val_g, val_nfeat, val_labels, val_nid, device)
	# 	print('Eval Acc {:.4f}'.format(eval_acc))
	# 	test_acc = evaluate(model, test_g, test_nfeat, test_labels, test_nid, device)
	# 	print('Test Acc: {:.4f}'.format(test_acc))
	print('Avg epoch time: {}'.format(avg / (epoch - 4)))
	print(time1_list)
	print('avg epoch GPU data transfer time:%.8f' % (sum(time1_list[6:]) / len(time1_list[6:])))


# print('Avg epoch time: {}'.format(avg / (epoch - 1)))


# current, peak = tracemalloc.get_traced_memory()
# print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
# tracemalloc.stop()

if __name__=='__main__':
	# get_memory("-----------------------------------------main_start***************************")
	tt = time.time()
	print("main start at this time " + str(tt))
	argparser = argparse.ArgumentParser("multi-gpu training")
	argparser.add_argument('--gpu', type=int, default=0,
		help="GPU device ID. Use -1 for CPU training")

	argparser.add_argument('--dataset', type=str, default='ogbn-products')

	argparser.add_argument('--num-epochs', type=int, default=2)
	argparser.add_argument('--num-hidden', type=int, default=16)
	argparser.add_argument('--num-layers', type=int, default=2)
	argparser.add_argument('--fan-out', type=str, default='10,25')

	# argparser.add_argument('--batch-size', type=int, default=196615)
	# argparser.add_argument('--batch-size', type=int, default=98308)
	# argparser.add_argument('--batch-size', type=int, default=49154)
	# argparser.add_argument('--batch-size', type=int, default=24577)
	# argparser.add_argument('--batch-size', type=int, default=12289)
	# argparser.add_argument('--batch-size', type=int, default=6145)
	argparser.add_argument('--batch-size', type=int, default=3000)
	# argparser.add_argument('--batch-size', type=int, default=1500)

	argparser.add_argument('--log-every', type=int, default=10)
	argparser.add_argument('--eval-every', type=int, default=5)
	argparser.add_argument('--lr', type=float, default=0.003)
	argparser.add_argument('--dropout', type=float, default=0.5)
	argparser.add_argument('--num-workers', type=int, default=4,
		help="Number of sampling processes. Use 0 for no extra process.")
	argparser.add_argument('--inductive', action='store_true',
		help="Inductive learning setting")
	argparser.add_argument('--data-cpu', action='store_true',
		help="By default the script puts all node features and labels "
		     "on GPU when using it to save time for data copy. This may "
		     "be undesired if they cannot fit in GPU memory at once. "
		     "This flag disables that.")
	args = argparser.parse_args()

	if args.gpu >= 0:
		device = th.device('cuda:%d' % args.gpu)
	else:
		device = th.device('cpu')
	# get_memory("-----------------------------------------before load_ogb***************************")
	t2 = ttt(tt, "before load_ogb")
	if args.dataset=='reddit':
		g, n_classes = load_reddit()
	if args.dataset=='ogbn-products':
		g, n_classes = load_ogb(args.dataset)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
	# get_memory("-----------------------------------------after load_ogb***************************")

	# if args.dataset in ['arxiv', 'collab', 'citation', 'ddi', 'protein', 'ppa', 'reddit.dgl','products']:
	#     g, n_classes = load_data(args.dataset)
	else:
		raise Exception('unknown dataset')
	# see_memory_usage("-----------------------------------------after data to cpu------------------------")
	t3 = ttt(t2, "after load_ogb")
	if args.inductive:
		train_g, val_g, test_g = inductive_split(g)
		train_nfeat = train_g.ndata.pop('features')
		val_nfeat = val_g.ndata.pop('features')
		test_nfeat = test_g.ndata.pop('features')
		train_labels = train_g.ndata.pop('labels')
		val_labels = val_g.ndata.pop('labels')
		test_labels = test_g.ndata.pop('labels')

	else:
		train_g = val_g = test_g = g
		train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('features')
		train_labels = val_labels = test_labels = g.ndata.pop('labels')
	# get_memory("-----------------------------------------after inductive else***************************")
	t4 = ttt(t3, "after inductive else")

	if not args.data_cpu:
		train_nfeat = train_nfeat.to(device)
		train_labels = train_labels.to(device)
	# get_memory("-----------------------------------------after label***************************")
	t5 = ttt(t4, "after label")
	# Create csr/coo/csc formats before launching training processes with multi-gpu.
	# This avoids creating certain formats in each sub-process, which saves momory and CPU.
	train_g.create_formats_()
	# get_memory("-----------------------------------------after  train_g.create_formats_()***************************")
	val_g.create_formats_()
	# get_memory("-----------------------------------------after  train_g.create_formats_()***************************")
	test_g.create_formats_()
	# get_memory("-----------------------------------------before pack data***************************")
	t6 = ttt(t5, "after train_g.create_formats_()")
	# see_memory_usage("-----------------------------------------after model to gpu------------------------")
	# Pack data
	data = n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
	       val_nfeat, val_labels, test_nfeat, test_labels
	# get_memory("-----------------------------------------after pack data***************************")
	t7 = ttt(t6, "after pack data")
	run(args, device, data, t6)
