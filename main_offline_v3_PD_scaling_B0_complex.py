from MRF.Offline.Network_v3_PD_scaling_B0_complex import *
from MRF.Training_parameters import *
from MRF.Offline.Data_class import *
from MRF.Projection_complex import *
import pickle
import argparse
import os


def load_parser():
	parser = argparse.ArgumentParser(description='Description of the training parameters.')
	parser.add_argument('-f','--save_name', type=str)
	# the prjection
	parser.add_argument('--project_lr_times', type=int, default=1)

	args = parser.parse_args()
	with open('settings_files_offline/settings_'+args.save_name+'.pkl', 'rb') as f:
		default_settings = pickle.load(f)
		parser.add_argument('--model', type=str, default=default_settings['model'])
		parser.add_argument('--optimizer', type=str, default=default_settings['optimizer'])
		parser.add_argument('--lr', type=float, default=default_settings['lr'])
		parser.add_argument('--noise_type', type=str, default=default_settings['noise_type'])
		parser.add_argument('--noise_level', type=float, default=default_settings['noise_level'])
		parser.add_argument('--normalization', type=str, default=default_settings['normalization'])
		parser.add_argument('--namepca', type=str, default=default_settings['namepca'])
		parser.add_argument('--batch_size', type=int, default=default_settings['batch_size'])
		parser.add_argument('--start_by_projection', type=bool, default=default_settings['start_by_projection'])
		parser.add_argument('--nb_epochs', type=int, default=default_settings['nb_epochs'])
		parser.add_argument('--initialization', type=str, default=default_settings['initialization'])
		parser.add_argument('--params', metavar='N', type=int, nargs='+', default=default_settings['params'])
		parser.add_argument('--loss', type=str, default=default_settings['loss'])
		parser.add_argument('--minPD', type=float, default=default_settings['minPD'])
		parser.add_argument('--maxPD', type=float, default=default_settings['maxPD'])
		parser.add_argument('--validation', type=bool, default=default_settings['validation'])
		parser.add_argument('--small_validation_size', type=int, default=default_settings['small_validation_size'])
		parser.add_argument('--validation_size', type=int, default=default_settings['validation_size'])
		parser.add_argument('--dimension_projection', type=float, default=default_settings['dimension_projection'])
		parser.add_argument('--nb_files', type=int, default=default_settings['nb_files'])		
		parser.add_argument('--path_files', type=str, default=default_settings['path_files'])
		return parser



def main():
	parser = load_parser()
	args = parser.parse_args()	
	
	training_parameters = Training_parameters(args.batch_size, 1, args.nb_epochs, args.params, args.normalization)
	
	CRBrequired = False
	for para in args.params:
		CRBrequired = CRBrequired or Loss.CRBrequired(args.loss[para])
	data_class = Data_class(training_parameters, args.noise_type, args.noise_level, args.minPD, args.maxPD, args.nb_files, args.path_files, CRBrequired = CRBrequired)
	if args.start_by_projection:
		projection = Projection(args.start_by_projection, args.dimension_projection, args.initialization, args.normalization, args.namepca)
	else:
		projection = None

	validation_settings = {'validation': args.validation,'small_validation_size': args.small_validation_size, 'validation_size': args.validation_size}
	network = Network_v3(args.model, args.loss, training_parameters, args.save_name, data_class, validation_settings, projection=projection)
	network.train(lr = args.lr, nameoptimizer = args.optimizer, projection_lr_times=args.project_lr_times)


if __name__ == '__main__':
	main()
