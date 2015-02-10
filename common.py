import csv
import gzip
from sets import Set

class Molecule:
	"""
	Molecule object


	Initialize with row read from provided csv files
	Contains:
		smiles: SMILES string representation of molecule
		features: 256-element list of binary features
		gap: HOMO-LUMO gap value
	"""

	def __init__(self, csv_row):
		self.smiles = csv_row[0]
		self.features = [int(float(item)) for item in row[1:-1]]
		self.gap = float(csv_row[-1])

def load_molecules(file_name="train.csv.gz"):
	"""Returns a list of molecule objects"""
	with gzip.open(file_name) as train_file:
		train_reader = csv.reader(train_file)
		train_reader.next()

		molecules = []
		for index, row in enumerate(train_reader):
			molecules.append(Molecule(row))

def to_fann_file(molecule_list, output_file, inputs, outputs):
	with gzip.open(output_file) as out_file:
		outfile.write("%s %s %s" % (len(molecule_list), inputs, outputs))
		for molecule in molecule_list:
			outfile.write(molecule.features.join(' '))
			outfile.write(molecule.gap)