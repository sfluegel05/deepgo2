import math
import pickle
import sys
from functools import partial
from importlib import import_module
from multiprocessing import Pool
from pathlib import Path

import click as ck
import numpy as np
import torch as th
from torch.nn import functional as F

from deepgo.data import load_data, load_normal_forms
from deepgo.metrics import compute_roc
from deepgo.models import DeepGOModel
from deepgo.utils import Ontology, propagate_annots
from deepgo.torch_utils import FastTensorDataLoader

MODEL_CHOICES = [
	"deepgozero",
	"deepgozero_plus",
	"deepgozero_esm",
	"deepgozero_esm_plus",
]


def infer_model_name(model_file: str) -> str:
	stem = Path(model_file).stem
	# Match longer names first so "deepgozero_esm_plus" does not resolve to "deepgozero_esm".
	for candidate in sorted(MODEL_CHOICES, key=len, reverse=True):
		if stem.startswith(candidate):
			return candidate
	raise ck.UsageError(
		"Could not infer --model-name from --model-file. Provide --model-name explicitly."
	)


@ck.command()
@ck.option("--data-root", "-dr", default="data", help="Data folder")
@ck.option(
	"--ont",
	"-ont",
	default="mf",
	type=ck.Choice(["mf", "bp", "cc"]),
	help="GO subontology",
)
@ck.option(
	"--model-file",
	"-mf",
	required=True,
	type=ck.Path(exists=True, dir_okay=False, path_type=Path),
	help="Path to model weights (.th)",
)
@ck.option(
	"--model-name",
	"-m",
	required=False,
	type=ck.Choice(MODEL_CHOICES),
	help="Model architecture name; inferred from model filename when omitted",
)
@ck.option(
	"--test-data-name",
	"-td",
	default="test",
	type=ck.Choice(["test", "nextprot", "valid"]),
	help="Test data set name",
)
@ck.option("--batch-size", "-bs", default=37, help="Batch size for evaluation")
@ck.option("--device", "-d", default="cuda:0", help="Device")
@ck.option(
	"--metrics-file",
	required=False,
	type=ck.Path(dir_okay=False, path_type=Path),
	help="Optional path to write evaluation metrics (.pf)",
)
@ck.option(
	"--output-file",
	required=False,
	type=ck.Path(dir_okay=False, path_type=Path),
	help="Optional path to write propagated predictions (.pkl)",
)
@ck.option(
	"--propagation-workers",
	default=32,
	show_default=True,
	type=int,
	help="Number of worker processes used for ontology propagation",
)
def main(
	data_root,
	ont,
	model_file,
	model_name,
	test_data_name,
	batch_size,
	device,
	metrics_file,
	output_file,
	propagation_workers,
):
	"""Standalone test evaluation for DeepGO models."""

	model_file = Path(model_file)
	model_stem = model_file.stem
	if model_name is None:
		model_name = infer_model_name(str(model_file))

	if device.startswith("cuda") and not th.cuda.is_available():
		print(
			f"WARNING: Requested device {device}, but CUDA is not available in this "
			"PyTorch build. Falling back to cpu."
		)
		device = "cpu"

	if model_name.find("plus") != -1:
		go_norm_file = f"{data_root}/go-plus.norm"
	else:
		go_norm_file = f"{data_root}/go.norm"

	go_file = f"{data_root}/go.obo"
	terms_file = f"{data_root}/{ont}/terms.pkl"
	if metrics_file is None:
		metrics_file = Path(f"{data_root}/{ont}/valid_{model_stem}.pf")
	if output_file is None:
		output_file = Path(f"{data_root}/{ont}/{test_data_name}_predictions_{model_stem}.pkl")

	go = Ontology(go_file, with_rels=True)

	if model_name.find("esm") != -1:
		features_length = 2560
		features_column = "esm2"
	else:
		features_length = None
		features_column = "interpros"

	test_data_file = f"{test_data_name}_data.pkl"
	iprs_dict, terms_dict, train_data, valid_data, test_data, test_df = load_data(
		data_root,
		ont,
		terms_file,
		features_length,
		features_column,
		test_data_file=test_data_file,
	)
	n_terms = len(terms_dict)
	if features_column == "interpros":
		features_length = len(iprs_dict)

	_, valid_labels = valid_data
	_, test_labels = test_data
	valid_labels = valid_labels.detach().cpu().numpy()
	test_labels = test_labels.detach().cpu().numpy()

	_, _, _, _, relations, zero_classes = load_normal_forms(go_norm_file, terms_dict)
	n_rels = len(relations)
	n_zeros = len(zero_classes)

	valid_loader = FastTensorDataLoader(*valid_data, batch_size=batch_size, shuffle=False)
	test_loader = FastTensorDataLoader(*test_data, batch_size=batch_size, shuffle=False)

	net = DeepGOModel(features_length, n_terms, n_zeros, n_rels, device).to(device)
	print("Loading the model weights")
	net.load_state_dict(th.load(model_file, map_location=device))
	net.eval()

	with th.no_grad():
		valid_steps = int(math.ceil(len(valid_labels) / batch_size))
		valid_loss = 0.0
		valid_preds = []
		with ck.progressbar(length=valid_steps, show_pos=True) as bar:
			for batch_features, batch_labels in valid_loader:
				bar.update(1)
				batch_features = batch_features.to(device)
				batch_labels = batch_labels.to(device)
				logits = net(batch_features)
				batch_loss = F.binary_cross_entropy(logits, batch_labels)
				valid_loss += batch_loss.detach().item()
				valid_preds.append(logits.detach().cpu().numpy())
		valid_loss /= valid_steps
	valid_preds = np.concatenate(valid_preds)
	valid_preds = list(valid_preds)
	with Pool(propagation_workers) as p:
		valid_preds = p.map(partial(propagate_annots, go=go, terms_dict=terms_dict), valid_preds)
	pickle.dump(valid_preds, open(metrics_file.with_suffix(".valid_preds.pkl"), "wb"))

	with th.no_grad():
		test_steps = int(math.ceil(len(test_labels) / batch_size))
		test_loss = 0.0
		preds = []
		with ck.progressbar(length=test_steps, show_pos=True) as bar:
			for batch_features, batch_labels in test_loader:
				bar.update(1)
				batch_features = batch_features.to(device)
				batch_labels = batch_labels.to(device)
				logits = net(batch_features)
				batch_loss = F.binary_cross_entropy(logits, batch_labels)
				test_loss += batch_loss.detach().cpu().item()
				preds.append(logits.detach().cpu().numpy())
		test_loss /= test_steps

	preds = np.concatenate(preds)
	roc_auc = compute_roc(test_labels, preds)
	print(f"Valid Loss - {valid_loss}, Test Loss - {test_loss}, Test AUC - {roc_auc}")

	metrics_file.parent.mkdir(parents=True, exist_ok=True)
	with open(metrics_file, "w") as f:
		f.write(f"Valid Loss - {valid_loss}, Test Loss - {test_loss}, Test AUC - {roc_auc}\n")

	preds = list(preds)
	with Pool(propagation_workers) as p:
		preds = p.map(partial(propagate_annots, go=go, terms_dict=terms_dict), preds)
	pickle.dump(preds, open(metrics_file.with_suffix(".test_preds.pkl"), "wb"))

	test_df["preds"] = preds
	output_file.parent.mkdir(parents=True, exist_ok=True)
	test_df.to_pickle(output_file)
	print(f"Saved propagated predictions to {output_file}")


if __name__ == "__main__":
	main()
