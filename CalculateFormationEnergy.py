#!/usr/bin/env python3
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
from Tools.AIMPROIOTools import get_output_file_name, get_final_energy, find_pristine_directory, count_atoms
from Tools.consts import EV_PER_AU

class Species:
	# Class to define a species in the system defined in the object name.
	def __init__(self, name, species_index = -1):
		self.name = name # elemental name e.g. C, Ga, etc
		self.species_index = species_index # species index in the given AIMPRO calculation
		self.count = 0 # number of this species in the system
		self.chemical_potential_eV = 0.0 # the chemical potential for this species if it is independent of the growth condition.
		self.chemical_potential_rich_eV = 0.0 # the rich chemical potential if it is dependent on growth condition.
		self.chemical_potential_lean_eV = 0.0 # the lean chemical potential if it is dependent on growth condition.
	def print_species(self):
		print(self.name)
		print(self.count)
		#print(self.chemical_potential_eV)
		#print(self.chemical_potential_rich_eV)
		#print(self.chemical_potential_lean_eV)
		print("") # just for a new line statement

class FormationEnergy:
	# Class to store formation energy data.
	def __init__(self, species_name):
		self.supercell_size = 0 # supercell dimension nxnx1
		self.inverse_supercell_size = 0.0 # inverse of the supercell dimension
		self.leanrich_species_name = species_name # name of the species which is leanrich_species_name rich / lean
		self.rich_eV = 0.0 # formation energy in the leanrich_species_name rich condition
		self.lean_eV = 0.0 # formation energy in the leanrich_species_name lean condition

def get_species(output_file_path):
	"""
	Counts how many atoms of each species are in a the output file at the specified path. 
	
	Args: output_file_path (str)
	Returns: species_list (list of Species objects)
	"""
	def parse_species_symbol(line):
		match = re.search(r'pot=\d+-([A-Z][a-z]*)', line) # Regular expression to capture only the element symbol after "pot="
		if match:
			species_symbol = match.group(1)
			return species_symbol
		else:
			raise ValueError("Input string format is incorrect")
	
	def populate_species(species_list, lines):
		positions_section_flag = False
		for line in lines:
			if "begin" in line and "{positions}" in line:
				positions_section_flag = True
			elif "end{positions}" in line:
				break
			elif positions_section_flag and line.strip() and not line.strip().startswith("!"): ######### NEED TO ACCOUNT FOR COMMENTED OUT ATOMS
				parts = line.split()
				if int(parts[1]) >= 1 and int(parts[1]) <= len(species_list):
					species_list[int(parts[1]) - 1].count += 1 # -1 accounts for the counting difference between the species list in the script and in AIMPRO
				else:
					raise ValueError("Format is incorrect. Species index is out of range.")
	
	with open(output_file_path, 'r') as output_file:
		lines = output_file.readlines()
	
	species_list = []
	species_section_flag = False
	species_index = 0
	for line in lines:
		if "begin{hghpseudo}" in line:
			species_section_flag = True
		elif "end{hghpseudo}" in line:
			break
		elif species_section_flag and line.strip(): # ignores blank lines 
			species_index += 1
			species_symbol = parse_species_symbol(line)
			individual_species = Species(species_symbol, species_index=species_index)
			species_list.append(individual_species)
	
	populate_species(species_list, lines)
	return species_list

def get_pristine_and_defective_species_lists(pristine_output_file_path, defective_output_file_path):
	"""
	Parses the defect's output file and the corresponding pristine output to count how many atoms of each species are in them. 
	
	Args: pristine_output_file_path (str) : path to pristine output file
	      defective_output_file_path (str) : path to defective output file
	Returns:
	      pristine_species_list (list of Species objects) : all the species in the pristine system, counted, named, and indexed. Also includes the defective system's species counted with 0 atoms (if they aren't in the pristine system).
	      defective_species_list (list of Species objects) : all of the species in the defective system, counted, named and indexed. Also includes the pristine system's species counted with 0 atoms (if they aren't in the defective system).
	"""

	pristine_species_list = get_species(pristine_output_file_path)
	defective_species_list = get_species(defective_output_file_path)

	# In general, the defective species list may contain more species than the pristine, or indeed vice versa in the most general case. Therefore the species lists should be compared against each other and the missing species should be added with a count of zero
	# Look through defective species to see if they are in the pristine system
	for defective_species in defective_species_list:
		is_defective_species_in_pristine_species_list = False
		for pristine_species in pristine_species_list:
			if defective_species.name == pristine_species.name:
				is_defective_species_in_pristine_species_list = True
				continue
		if not is_defective_species_in_pristine_species_list:
			new_species = Species(defective_species.name, species_index=len(pristine_species_list) + 1)
			pristine_species_list.append(new_species)
	
	# Look through pristine species to see if they are in the defective system
	for pristine_species in pristine_species_list:
		is_pristine_species_in_defective_species_list = False
		for defective_species in defective_species_list:
			if pristine_species.name == defective_species.name:
				is_pristine_species_in_defective_species_list = True
				continue
		if not is_pristine_species_in_defective_species_list:
			new_species = Species(pristine_species.name, species_index=len(defective_species_list) + 1)
			defective_species_list.append(new_species)

	return pristine_species_list, defective_species_list

def write_to_data_file(formation_energies, imbalanced_species_list, pristine_species_for_lean_calculation_list, c_rich, c_lean):
	output_file_name = "formation_energy_data_file"
	with open(output_file_name, "w") as output_file:
		output_file.write("begin{chemical_potentials}[eV]\n")
		for imbalanced_species in imbalanced_species_list:
			if imbalanced_species.chemical_potential_eV != 0.0:
				output_file.write(f"{imbalanced_species.name} = {imbalanced_species.chemical_potential_eV}\n")
			elif imbalanced_species.chemical_potential_rich_eV != 0.0:
				output_file.write(f"{imbalanced_species.name} ({formation_energies[0].leanrich_species_name}-rich) = {imbalanced_species.chemical_potential_rich_eV}\n")
				output_file.write(f"{imbalanced_species.name} ({formation_energies[0].leanrich_species_name}-lean) = {imbalanced_species.chemical_potential_lean_eV}\n")
		for pristine_species in pristine_species_for_lean_calculation_list:
			output_file.write(f"{pristine_species.name} = {pristine_species.chemical_potential_rich_eV}\n")
		output_file.write("end{chemical_potentials}\n")
		output_file.write("\n")
		if formation_energies[0].leanrich_species_name == "placeholder name": # this occurs when there is no rich-lean dependence
			output_file.write(f"Supercell Size, Formation Energy (eV)\n")
			output_file.write("begin{data}\n")
			for formation_energy in formation_energies:
				output_file.write(f"{formation_energy.supercell_size}, {formation_energy.rich_eV}\n")
			if c_rich:
				output_file.write(f"Dilute, {c_rich}\n")
		else:
			output_file.write(f"Supercell Size, Formation Energy in {formation_energies[0].leanrich_species_name}-rich limit (eV), Formation Energy in {formation_energies[0].leanrich_species_name}-lean limit (eV)\n")
			output_file.write("begin{data}\n")
			for formation_energy in formation_energies:
				output_file.write(f"{formation_energy.supercell_size}, {formation_energy.rich_eV}, {formation_energy.lean_eV}\n")
			if c_rich and c_lean:
				output_file.write(f"Dilute, {c_rich}, {c_lean}\n")
		output_file.write("end{data}\n")
	print(f"Data written to {os.path.join(os.getcwd(), output_file_name)}")

def plot_graph(formation_energies, m_rich, m_lean, c_rich, c_lean):
	plt.rcParams['text.usetex'] = True
	plt.rcParams['font.family'] = 'serif'

	# RICH GRAPH
	fig_inverseSupercellSize_rich_formationEnergy, axis_inverseSupercellSize_rich_formationEnergy = plt.subplots(figsize=(12,10)) # values are in inches, default is 6.4,4.8
	axis_inverseSupercellSize_rich_formationEnergy.scatter([formation_energy.inverse_supercell_size for formation_energy in formation_energies], [formation_energy.rich_eV for formation_energy in formation_energies], color="#1f77b4", marker='x', s=140)

	# Add regression line
	if m_rich and c_rich:
		axis_inverseSupercellSize_rich_formationEnergy.plot([formation_energy.inverse_supercell_size for formation_energy in formation_energies], m_rich * np.array([formation_energy.inverse_supercell_size for formation_energy in formation_energies]) + c_rich, color='black', label=f'Linear Regression\n$y$ = {m_rich:.4f}$x$ + {c_rich:.4f} eV')
	
	axis_inverseSupercellSize_rich_formationEnergy.set_xlabel('Inverse of supercell dimension, 1/n', fontsize=24)
	axis_inverseSupercellSize_rich_formationEnergy.set_ylabel('Formation Energy (eV)', fontsize=24)
	axis_inverseSupercellSize_rich_formationEnergy.yaxis.set_tick_params(labelsize=24)
	axis_inverseSupercellSize_rich_formationEnergy.xaxis.set_tick_params(labelsize=24)

	axis_inverseSupercellSize_rich_formationEnergy.legend(fontsize=24)
	if formation_energies[0].leanrich_species_name == "placeholder name":
		fig_inverseSupercellSize_rich_formationEnergy.savefig('inverseSupercellSize_formationEnergy', bbox_inches='tight')
	else:
		fig_inverseSupercellSize_rich_formationEnergy.savefig(f'inverseSupercellSize_{formation_energies[0].leanrich_species_name}_rich_formationEnergy', bbox_inches='tight')
	
	# LEAN GRAPH
	if formation_energies[0].leanrich_species_name != "placeholder name":
		fig_inverseSupercellSize_lean_formationEnergy, axis_inverseSupercellSize_lean_formationEnergy = plt.subplots(figsize=(12,10)) # values are in inches, default is 6.4,4.8
		axis_inverseSupercellSize_lean_formationEnergy.scatter([formation_energy.inverse_supercell_size for formation_energy in formation_energies], [formation_energy.lean_eV for formation_energy in formation_energies], color="#1f77b4", marker='x', s=140)
		
		# Add regression line
		if m_lean and c_lean:
			axis_inverseSupercellSize_lean_formationEnergy.plot([formation_energy.inverse_supercell_size for formation_energy in formation_energies], m_lean * np.array([formation_energy.inverse_supercell_size for formation_energy in formation_energies]) + c_lean, color='black', label=f'Linear Regression\n$y$ = {m_lean:.4f}$x$ + {c_lean:.4f} eV')
		
		axis_inverseSupercellSize_lean_formationEnergy.set_xlabel('Inverse of supercell dimension, 1/n', fontsize=24)
		axis_inverseSupercellSize_lean_formationEnergy.set_ylabel('Formation Energy (eV)', fontsize=24)
		axis_inverseSupercellSize_lean_formationEnergy.yaxis.set_tick_params(labelsize=24)
		axis_inverseSupercellSize_lean_formationEnergy.xaxis.set_tick_params(labelsize=24)

		axis_inverseSupercellSize_lean_formationEnergy.legend(fontsize=24)
		fig_inverseSupercellSize_lean_formationEnergy.savefig(f'inverseSupercellSize_{formation_energies[0].leanrich_species_name}_lean_formationEnergy', bbox_inches='tight')

parser = argparse.ArgumentParser(description="Calculate formation energy of the defect in the pwd.")
parser.add_argument("--pristine_directories_path", "-pristinePath", type=str, default=None,
                    help="=Specify the path to the directory where the different pristine cell size directories are held. If left blank, the default calculation will be located.")
# THESE ARE NOT IMPLEMENTED YET
#parser.add_argument("--one_shot_flag", "-oneShot", action="store_true",  # Becomes True if specified, otherwise False
                    #help="Swap to just a single calculation of formation energy.")


args = parser.parse_args()

pwd = os.getcwd()
defective_supercell_relative_directories = [
	d for d in os.listdir(pwd)
	if os.path.isdir(os.path.join(pwd, d)) and re.fullmatch(r"\d+x\d+", d)
]

print(defective_supercell_relative_directories)
if not args.pristine_directories_path:
	pristine_directories_path = find_pristine_directory(pwd)
else:
	pristine_directories_path = args.pristine_directories_path


formation_energies = []

for defective_supercell_relative_directory in defective_supercell_relative_directories:
	formation_energy = FormationEnergy("placeholder name")
	# get supercell size from directory path to obtain the appropriate pristine path.
	supercell_size_match = re.search(r"(\d+)x(\d+)", defective_supercell_relative_directory)
	if supercell_size_match:
		pristine_directory_path = os.path.join(pristine_directories_path, "EnergyCalculation", defective_supercell_relative_directory)
		formation_energy.supercell_size = int(supercell_size_match.group(1))
	else:
		raise ValueError("Incorrectly formatted supercell directory. No /nxn/")

	# Get Pristine Energy
	pristine_output_file_name = get_output_file_name(pristine_directory_path)
	pristine_output_file_path = os.path.join(pristine_directory_path, pristine_output_file_name)
	pristine_energy_eV = get_final_energy(pristine_output_file_path) * EV_PER_AU
	
	# Get Defective Energy
	defective_supercell_absolute_directory_path = os.path.abspath(defective_supercell_relative_directory)
	defective_output_file_name = get_output_file_name(defective_supercell_absolute_directory_path)
	defective_output_file_path = os.path.join(defective_supercell_absolute_directory_path, defective_output_file_name)
	defective_energy_eV = get_final_energy(defective_output_file_path) * EV_PER_AU
	
	pristine_species_list, defective_species_list = get_pristine_and_defective_species_lists(pristine_output_file_path, defective_output_file_path)

	# Identify differences in numbers of species 
	imbalanced_species_list = []
	for defective_species in defective_species_list:
		for pristine_species in pristine_species_list:
			if defective_species.name == pristine_species.name:
				corresponding_number_of_atoms_in_pristine_system = pristine_species.count
		species_difference = defective_species.count - corresponding_number_of_atoms_in_pristine_system
		if species_difference != 0:
			imbalanced_species = Species(defective_species.name)
			imbalanced_species.count = species_difference
			imbalanced_species_list.append(imbalanced_species)
	print("Imbalanced species:")
	for species in imbalanced_species_list:
		species.print_species()
	
	# Now get the corresponding chemical potentials. This is only for the rich conditions. Lean is calculated later
	for imbalanced_species in imbalanced_species_list:
		solid_reference_directory_path = os.path.join("/home/c2047921/SolidReferences/", imbalanced_species.name)
		solid_reference_output_file_name = get_output_file_name(solid_reference_directory_path)
		solid_reference_output_file_path = os.path.join(solid_reference_directory_path, solid_reference_output_file_name)
		solid_reference_final_energy_eV = get_final_energy(solid_reference_output_file_path) * EV_PER_AU
		
		# Count how many atoms in system
		num_atoms_rich = count_atoms(solid_reference_output_file_path)
		chemical_potential_eV = solid_reference_final_energy_eV / num_atoms_rich
		
		# Now we need to check if the imbalanced species is in the pristine system. If it is we need to specify that the chemical potential is only in the rich case, rather than just the general chemical potentials.
		is_imbalanced_species_in_pristine_system = False
		for pristine_species in pristine_species_list:
			if pristine_species.count == 0:
				continue
			elif imbalanced_species.name == pristine_species.name:
				formation_energy.leanrich_species_name = imbalanced_species.name
				is_imbalanced_species_in_pristine_system = True
				break
		
		if is_imbalanced_species_in_pristine_system:
			imbalanced_species.chemical_potential_rich_eV = chemical_potential_eV
		else:
			imbalanced_species.chemical_potential_eV = chemical_potential_eV
	
	# Calculate lean chemical potentials
	pristine_species_for_lean_calculation_list = []
	pristine_primitive_directory_path = os.path.join(pristine_directories_path, "EnergyCalculation", "Primitive")
	pristine_primitive_output_file_path = os.path.join(pristine_primitive_directory_path, get_output_file_name(pristine_primitive_directory_path))
	pristine_primitive_species_list = get_species(pristine_primitive_output_file_path)
	pristine_primitive_energy_eV = get_final_energy(pristine_primitive_output_file_path) * EV_PER_AU
	for imbalanced_species in imbalanced_species_list:
		if imbalanced_species.chemical_potential_rich_eV != 0.0: # this selects all atoms which need their lean chemical potentials calculated.
			imbalanced_species_lean_chemical_potential_eV = pristine_primitive_energy_eV
			# find the energy per atom in that system by subtracting the energy of each of the other types of atoms in that system from the total, normalising for one atom
			for pristine_primitive_species in pristine_primitive_species_list:
				if pristine_primitive_species.name == imbalanced_species.name: # this excludes the species for which we are calculating the lean chemical potential. If we didn't we'd simply get 0 eV.
					lean_species_count = pristine_primitive_species.count
					continue
				 # get chemical potential for this species
				solid_reference_directory_path = os.path.join("/home/c2047921/SolidReferences/", pristine_primitive_species.name)
				solid_reference_output_file_name = get_output_file_name(solid_reference_directory_path)
				solid_reference_output_file_path = os.path.join(solid_reference_directory_path, solid_reference_output_file_name)
				num_atoms_lean = count_atoms(solid_reference_output_file_path)
				pristine_species_chemical_potential_rich_eV = get_final_energy(solid_reference_output_file_path) * EV_PER_AU / num_atoms_lean
				pristine_species_for_lean_calculation = Species(pristine_primitive_species.name)
				pristine_species_for_lean_calculation.chemical_potential_rich_eV = pristine_species_chemical_potential_rich_eV
				pristine_species_for_lean_calculation_list.append(pristine_species_for_lean_calculation)
				imbalanced_species_lean_chemical_potential_eV -= pristine_species_for_lean_calculation.chemical_potential_rich_eV * pristine_primitive_species.count
			imbalanced_species.chemical_potential_lean_eV = imbalanced_species_lean_chemical_potential_eV/lean_species_count
	
	# Calculate formation energies
	# Calculate rich correction
	species_correction_rich_eV = 0.0
	for imbalanced_species in imbalanced_species_list:
		if imbalanced_species.chemical_potential_eV != 0.0:
			species_correction_rich_eV += imbalanced_species.chemical_potential_eV * imbalanced_species.count
		elif imbalanced_species.chemical_potential_rich_eV != 0.0:
			species_correction_rich_eV += imbalanced_species.chemical_potential_rich_eV * imbalanced_species.count
	formation_energy.rich_eV = defective_energy_eV - pristine_energy_eV - species_correction_rich_eV
	
	# Calculate lean correction
	species_correction_lean_eV = 0.0
	for imbalanced_species in imbalanced_species_list:
		if imbalanced_species.chemical_potential_eV != 0.0:
			species_correction_lean_eV += imbalanced_species.chemical_potential_eV * imbalanced_species.count
		elif imbalanced_species.chemical_potential_lean_eV != 0.0:
			species_correction_lean_eV += imbalanced_species.chemical_potential_lean_eV * imbalanced_species.count
	formation_energy.lean_eV = defective_energy_eV - pristine_energy_eV - species_correction_lean_eV
	
	formation_energies.append(formation_energy)

if len(formation_energies) > 1:
	# fit linear for formation energies as a function of inverse supercell sizes
	for formation_energy in formation_energies:
		formation_energy.inverse_supercell_size = (formation_energy.supercell_size)**(-1)
	m_rich, c_rich = np.polyfit([formation_energy.inverse_supercell_size for formation_energy in formation_energies], [formation_energy.rich_eV for formation_energy in formation_energies], 1) 
	m_lean, c_lean = np.polyfit([formation_energy.inverse_supercell_size for formation_energy in formation_energies], [formation_energy.lean_eV for formation_energy in formation_energies], 1)
else:
	m_rich, c_rich, m_lean, c_lean = None, None, None, None

formation_energies.sort(key=lambda formation_energy: formation_energy.supercell_size)
write_to_data_file(formation_energies, imbalanced_species_list, pristine_species_for_lean_calculation_list, c_rich, c_lean)
plot_graph(formation_energies, m_rich, m_lean, c_rich, c_lean)

