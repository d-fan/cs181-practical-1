import csv
import gzip
import struct
from sets import Set
from array import array
try:
	from bitarray import bitarray
	BITARRAY = True
except ImportError:
	BITARRAY = False

"""
Usage:
>>> import common
>>> molecules = common.load_molecules()
>>> common.to_fann_file(molecules, "outputfile.fann")
"""

class Molecule:
	"""
	Molecule object


	Initialize with row read from provided csv files
	Contains:
		smiles: SMILES string representation of molecule
		features: 256-element list of binary features
		gap: HOMO-LUMO gap value
	"""

	def __init__(self, *args):
		if len(args) == 1:
			csv_row = args[0]
			self.smiles = csv_row[0]
			self.features = [int(float(item)) for item in csv_row[1:-1]]
			self.gap = float(csv_row[-1])
		elif len(args) == 3:
			self.smiles = args[0]
			self.features = args[1]
			self.gap = args[2]

def load_molecules(file_name="train.csv.gz", max_count = None):
	"""Returns a list of molecule objects"""
	count = 0
	molecules = []
	with gzip.open(file_name) as train_file:
		train_reader = csv.reader(train_file)
		train_reader.next()

		for index, row in enumerate(train_reader):
			molecules.append(Molecule(row))
			count += 1
			if count == max_count:
				return molecules
	return molecules

def save_compact(molecules, filename):
	if not BITARRAY:
		return
	with open(filename, 'wb') as outfile:
		for molecule in molecules:
			# Compact binary representation of bit list
			feature_bytes = bitarray(molecule.features).tobytes() 
			# Binary representation of float
			gap_bytes = struct.pack('f', molecule.gap)
			# Add together with SMILES at end
			out = "%s%s" % (feature_bytes, gap_bytes)
			outfile.write(out)

def load_compact(filename):
	if not BITARRAY:
		return []
	molecules = []
	with open(filename, 'rb') as infile:
		while True:
			features_bytes = infile.read(256 / 8)
			features = bitarray()
			features.frombytes(features_bytes)
			gap_bytes = infile.read(4)
			gap = struct.unpack('f', gap_bytes)
			molecule = Molecule("No SMILES representation", features.tolist(), gap)
			molecules.append(molecule)
	return molecules

def to_fann_file(molecule_list, outfile_name):
	with open(outfile_name, 'w') as outfile:
		n_samples = len(molecule_list)
		n_inputs = len(molecule_list[0].features)
		n_outputs = 1
		outfile.write("%s %s %s\n" % (n_samples, n_inputs, n_outputs))
		for molecule in molecule_list:
			outfile.write(' '.join([str(i) for i in molecule.features]) + '\n')
			outfile.write(str(molecule.gap) + '\n')