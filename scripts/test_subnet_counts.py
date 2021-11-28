import utils
import pprint
import argparse
def parse_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("--subnet-beta", action="store_true")
	ap.add_argument("--subnet-fixed", action="store_true")
	return ap.parse_args()

def show_models(subnet_beta=False, subnet_fixed=False):
	results = []
	if subnet_beta:
		print("Running for subnet-beta")
		cfg_function = utils.get_subnet_configs_beta
	elif subnet_fixed:
		print("Running for subnet-fixed")
		cfg_function = utils.get_subnet_configs_fixed
	else:
		return
	for num_models in utils.ENSEMBLE_COUNTS:
		config = cfg_function(num_models, base_hidden_size=768)[0]
		model = utils.extract_subnetwork_from_bert(**config)
		n_params = utils.get_param_count(model)
		param_ratio, total_ratio = utils.get_param_ratios(n_params, num_models)
		config['param_ratio'] = param_ratio
		config['total_ratio'] = total_ratio
		results.append(config)	
	pprint.pp(list(zip(utils.ENSEMBLE_COUNTS, results)))

if __name__== '__main__':
	args = parse_args()
	print(args)
	show_models(subnet_beta=args.subnet_beta)
	show_models(subnet_fixed=args.subnet_fixed)
