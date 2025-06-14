import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Crippen
from rdkit.Chem.Scaffolds import MurckoScaffold
import requests
import json
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - (Function: %(funcName)s)')
logger = logging.getLogger(__name__)

@dataclass
class MutationTarget:
    """Data class for EGFR mutation information"""
    name: str
    residue_number: int
    wild_type: str
    mutant: str
    binding_site_impact: str
    resistance_mechanism: str

class EGFRMutationDatabase:
    """Database of EGFR mutations and their characteristics"""
    
    def __init__(self):
        self.mutations = {
            'exon19del': MutationTarget(
                name='Exon 19 Deletion',
                residue_number=746,
                wild_type='ELREA',
                mutant='del',
                binding_site_impact='ATP pocket reorganization',
                resistance_mechanism='Altered binding affinity'
            ),
            'T790M': MutationTarget(
                name='T790M',
                residue_number=790,
                wild_type='T',
                mutant='M',
                binding_site_impact='Gatekeeper steric clash',
                resistance_mechanism='Blocks first-generation TKI binding'
            ),
            'L858R': MutationTarget(
                name='L858R',
                residue_number=858,
                wild_type='L',
                mutant='R',
                binding_site_impact='Activation loop stabilization',
                resistance_mechanism='Increased kinase activity'
            ),
            'C797S': MutationTarget(
                name='C797S',
                residue_number=797,
                wild_type='C',
                mutant='S',
                binding_site_impact='Loss of covalent binding site',
                resistance_mechanism='Prevents irreversible inhibitor binding'
            ),
            'G719X': MutationTarget(
                name='G719X',
                residue_number=719,
                wild_type='G',
                mutant='S/A/C',
                binding_site_impact='P-loop destabilization',
                resistance_mechanism='Altered ATP binding'
            )
        }
        logger.info("EGFRMutationDatabase initialized.")
    
    def get_mutation_info(self, mutation_key: str) -> Optional[MutationTarget]:
        """Get information about a specific mutation"""
        logger.debug(f"Retrieving info for mutation: {mutation_key}")
        return self.mutations.get(mutation_key)
    
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExternalMolecularGenerator:
    """Generate osimertinib-like molecular analogs with enhanced diversity and stability considerations."""

    def __init__(self, generator_type: str = "osimertinib_analogs", **kwargs):
        """
        Initializes the ExternalMolecularGenerator.

        Args:
            generator_type (str): Specifies the type of molecular generation.
                                  Currently only "osimertinib_analogs" is supported.
            **kwargs: Placeholder for future configuration options.
        """
        self.generator_type = generator_type
        
        # Full SMILES string for Osimertinib (AZD-9291)
        self.osimertinib_full_smiles = "COc1cc(Nc2nccc(c3cn(C)c4ccccc34)n2)c(cc1N(C)CCN(C)C)NC(=O)C=C"
        
        if self.generator_type == "osimertinib_analogs":
            logger.info("Using Osimertinib-like scaffold decoration for molecule generation, focusing on EGFR mutations.")
            logger.info(f"Full Osimertinib SMILES: {self.osimertinib_full_smiles}")
        else:
            logger.warning(f"Unsupported generator_type: {self.generator_type}. Defaulting to 'osimertinib_analogs'.")
            self.generator_type = "osimertinib_analogs"

        # Comprehensive core aromaticities for EGFR inhibitors
        self.core_aromaticities = {
            # ========== TRADITIONAL EGFR CORES ==========
            "pyrimidine_original": "c1nccc(n1)",
            "quinazoline": "c1cnc2c(c1)ncnc2",
            "pyrido_pyrimidine": "c1cnc2ncncc2c1", 
            "pyrrolo_pyrimidine": "c1ncc2nc(C)ncc2c1",
            "thieno_pyrimidine": "c1csc2nc(C)ncc2c1",
            "imidazo_pyrimidine": "c1nc2cnc(n2)c1",
            "pyrazolo_pyrimidine": "c1cc2c(cc1)cnc(n2)",
            
            # ========== RESISTANCE-COMBATING CORES ==========
            "quinoline_scaffold": "c1ccc2ncccc2c1",
            "isoquinoline_scaffold": "c1cnc2ccccc2c1",
            "pyridopyrazine": "c1cnc2nccnc2c1",
            "pyrimidopyrimidine": "c1nc2ncncnc2n1",
            "triazolopyrimidine": "c1nnc2ncncc2n1",
            "tetrazolopyrimidine": "c1nnnn2ncncc12",
            "oxazolopyrimidine": "c1oc2ncncc2n1",
            "thiazolopyrimidine": "c1sc2ncncc2n1",
            "benzimidazole": "c1ccc2nc[nH]c2c1",
            "benzothiazole": "c1ccc2nc(s2)c1",
            "benzoxazole": "c1ccc2nc(o2)c1",
            "purine_core": "c1nc2c(ncn2)n1",
            "pteridine": "c1nc2nccnc2nc1",
            "quinoxaline": "c1cnc2ncccc2c1",
            "phthalazine": "c1ccc2nncc2c1",
            "cinnoline": "c1ccc2nnccc2c1",
            "pyridazine_fused": "c1nnc2ccccc2c1",
            
            # ========== COVALENT WARHEAD COMPATIBLE CORES ==========
            "pyrimidine_cys797": "c1nc(C)c(cn1)",
            "quinazoline_acrylamide": "c1cnc2c(c1)nc(C=C)nc2",
            "pyrazolopyrimidine_vinyl": "c1cc2c(cc1)cnc(C=C)n2",
            "indazole_core": "c1ccc2nn(cc2c1)",
            "benzisoxazole": "c1ccc2c(c1)noc2",
            "pyrimidopyridazine": "c1nnc2ncncc2c1",
            
            # ========== ALLOSTERIC SITE TARGETING ==========
            "naphthalene": "c1ccc2ccccc2c1",
            "anthracene": "c1ccc2cc3ccccc3cc2c1", 
            "phenanthrene": "c1ccc2c(c1)ccc3ccccc32",
            "pyrene": "c1cc2ccc3cccc4ccc(c1)c2c34",
            "chrysene": "c1ccc2cc3ccc4ccccc4c3cc2c1", 
            "acridine": "c1ccc2cc3ccccc3nc2c1",
            "phenazine": "c1ccc2nc3ccccc3nc2c1",
            "carbazole": "c1ccc2c(c1)[nH]c3ccccc32",
            "dibenzofuran": "c1ccc2c(c1)oc3ccccc32",
            "dibenzothiophene": "c1ccc2c(c1)sc3ccccc32",
            
            # ========== NOVEL HETEROCYCLIC SYSTEMS ==========
            "pyrimidothiazole": "c1nc2scnc2nc1",
            "oxadiazolopyrimidine": "c1nnc(o1)c2ncncc2",
            "thiadiazolopyrimidine": "c1nnc(s1)c2ncncc2",
            "triazinopyrimidine": "c1nc2ncncnc2nc1",
            "pyrazinopyrimidine": "c1cnc2ncncc2n1",
            "pyrimidinopyrazole": "c1cc2ncncc2nn1",
            "benzotriazole": "c1ccc2nn[nH]c2c1",
            "indolizine": "c1ccn2cccc2c1",
            "imidazopyridine": "c1cc2cnc(n2)cc1",
            "pyrazolopyridine": "c1cc2ccn(n2)cc1",
            "isothiazolopyrimidine": "c1snc2ncncc2c1",
            "isoxazolopyrimidine": "c1onc2ncncc2c1",
            
            # ========== MACROCYCLIC INSPIRED CORES ==========
            "cyclotetradecaheptaene": "C1=CC=CC=CC=CC=CC=CC=CC=C1",
            "cyclohexadecaoctaene": "C1=CC=CC=CC=CC=CC=CC=CC=CC=CC=C1", 
            "cyclooctadecanonaene": "C1=CC=CC=CC=CC=CC=CC=CC=CC=CC=CC=C1",
            "porphyrin_core": "c1cc2cc3ccc(cc4ccc(cc5ccc(n1)c2n5)n4)n3",
            "corannulene": "c1cc2ccc3ccc4ccc5ccc6ccc1c7c2c3c4c5c67",
            
            # ========== SPIROCYCLIC CORES ==========
            "spiro_quinazoline": "C1(c2ncnc3ccccc23)CCCC1",
            "spiro_pyrimidine": "C1(c2ncncc2)CCC1",
            "spiro_indazole": "C1(c2ccc3nn(cc3c2))CCCC1",
            
            # ========== BRIDGED HETEROCYCLES ==========
            "quinuclidine_pyrimidine": "C1CCN2CCC1CC2c3ncncc3",
            "norbornane_quinazoline": "C1CC2CCC1C2c3cnc4ncncc4c3",
            "adamantyl_pyrimidine": "C1C2CC3CC(C1)CC(C2)(C3)c4ncncc4",
        }
        
        # ========== ENHANCED OSIMERTINIB FRAGMENTS ==========
        self.osimertinib_fragments = {
            # Original fragments
            "core_pyrimidine": "c1nccc(n1)",
            "methoxy_aniline": "COc1cc(cc1)N",
            "indole_substituent": "c1cn(C)c2ccccc12",
            "dimethylamino_chain": "N(C)CCN(C)C",
            "acrylamide_group": "NC(=O)C=C",
            
            # Resistance-fighting aniline variants
            "difluoro_methoxy_aniline": "COc1cc(F)c(F)cc1N",
            "trifluoromethyl_aniline": "CF3c1ccc(N)cc1",
            "cyano_methoxy_aniline": "COc1cc(CN)cc1N",
            "nitro_aniline": "O=N(=O)c1ccc(N)cc1",
            "dimethylamino_aniline": "N(C)(C)c1ccc(N)cc1",
            "hydroxyl_aniline": "Oc1ccc(N)cc1",
            "thiol_aniline": "Sc1ccc(N)cc1",
            "phosphonate_aniline": "P(=O)(O)(O)c1ccc(N)cc1",
            
            # Alternative heterocyclic substituents
            "benzothiophene_substituent": "c1cc2sccc2cc1",
            "benzofuran_substituent": "c1cc2occc2cc1",  
            "quinoline_substituent": "c1ccc2ncccc2c1",
            "isoquinoline_substituent": "c1cnc2ccccc2c1",
            "carbazole_substituent": "c1ccc2c(c1)[nH]c3ccccc32",
            "phenothiazine_substituent": "c1ccc2c(c1)sc3ccccc3n2",
            "acridine_substituent": "c1ccc2cc3ccccc3nc2c1",
            
            # Covalent warhead alternatives
            "vinyl_ketone": "C=CC(=O)",
            "chloroacetamide": "ClCC(=O)N",
            "bromoacrylamide": "BrC=CC(=O)N",
            "propargyl_amide": "C#CCC(=O)N",
            "epoxide_warhead": "C1CO1",
            "aziridine_warhead": "C1CN1",
            "alpha_cyano_acrylamide": "N#CC(=C)C(=O)N",
            "maleimide_warhead": "O=C1C=CC(=O)N1",
            "vinyl_sulfone": "C=CS(=O)(=O)",
            "nitrile_oxide": "C#N[O]",
            
            # Solubility enhancing groups
            "morpholine_chain": "N1CCOCC1",
            "piperazine_chain": "N1CCNCC1",
            "pyrrolidine_chain": "N1CCCC1",
            "imidazole_chain": "c1cnc[nH]1",
            "tetrazole_chain": "c1nnn[nH]1",
            "phosphate_chain": "OP(=O)(O)O",
            "sulfonate_chain": "S(=O)(=O)O",
            "carboxylate_chain": "C(=O)O",
            
            # Metabolically stable linkers
            "oxetane_linker": "C1COC1",
            "azetidine_linker": "C1CNC1",
            "spiro_linker": "C1(CCCC1)C2CC2",
            "gem_dimethyl": "C(C)(C)",
            "deuterated_methyl": "[2H]C([2H])([2H])",
            "fluorinated_linker": "CF2CF2",
            "silicon_linker": "[Si](C)(C)",
        }
        
        # ========== COMPREHENSIVE MODIFIERS ==========
        self.generation_2_modifiers = [
            # Traditional halogens and alkyl
            "F", "Cl", "Br", "I", "CH3", "CF3", "OH", "NH2", "CN", "OCH3", "OCHF2", "SCH3",
            
            # Resistance-fighting substituents
            "CHF2", "CF2CF3", "OCF3", "SCHF2", "SCF3", "SF5",
            "NO2", "SO2NH2", "SO2CH3", "PO3H2", "B(OH)2",
            "N(CH3)2", "NHCH3", "NHCF3", "N3", "ONH2",
            "SH", "SOH", "SO2H", "SeH", "TeH",
            "C#N", "C#CH", "C=CH2", "C#C-CH3",
            "CH2OH", "CH2NH2", "CH2CN", "CH2F", "CHF2", "CF2H",
            
            # Cyclic substituents
            "c1ccccc1", "c1ccncc1", "c1cncnc1", "C1CC1", "C1CCC1", "C1CCCC1",
            "c1ccoc1", "c1ccsc1", "c1cnn[nH]1", "c1cnoc1", "c1cnsc1",
            
            # Metabolically stable groups
            "[2H]", "[2H]C([2H])([2H])", "C(F)(F)F", "Si(CH3)3", "Ge(CH3)3",
            "P(=O)(OH)2", "As(=O)(OH)2", "S(=O)(=O)NH2", "Se(=O)(=O)NH2",
        ]
        
        # ========== ADVANCED LINKERS ==========
        self.generation_3_linkers = [
            # Traditional linkers
            "C(=O)N", "S(=O)2N", "OC(=O)", "c1ccccc1", "CONH", "SO2", "NHCO", "CH2CH2",
            
            # Resistance-bypassing linkers
            "CF2CF2", "CHF-CHF", "C#C", "C=C", "N=N", "N=C", "C=N",
            "OCF2", "SCF2", "NCF2", "PCF2", "SiF2", "GeF2",
            "c1cc(F)c(F)cc1", "c1cncc1", "c1cncnc1", "c1nccnc1",
            "C1CC1", "C1CCC1", "C1CCCC1", "C1CCCCC1",
            "C1COC1", "C1CNC1", "C1CSC1", "C1CPC1",
            "N1CCN(CC1)", "N1CCOCC1", "N1CCCC1", "N1CCCCC1",
            "S(=O)", "S(=O)2", "P(=O)", "As(=O)", "Se(=O)",
            "B-N", "Si-O", "Ge-S", "Sn-N", "Pb-O",
            
            # Conformationally restricted linkers
            "C1=CC=CC=C1", "c1ccc2ccccc2c1",
            "C1CC2CC1CC2", "C12CC3CC(C1)CC(C2)C3",
            "C1(CCCC1)", "C1(CCC1)", "C1(CC1)",
            
            # Bioisosteric replacements
            "C(=S)N", "C(=Se)N", "P(=O)N", "As(=O)N",
            "CONHSO2", "SO2NHCO", "CONHPO2", "PO2NHCO",
            "OCO", "SCS", "NCN", "PCP", "AsAs",
            "c1nn[nH]c1", "c1noc[nH]1", "c1nsc[nH]1",
            
            # Metabolically stable linkers 
            "[2H]C([2H])", "CF2", "SiF2", "GeF2", "SnF2",
            "C(C)(C)", "C(F)(F)", "C(Cl)(Cl)", "C(Br)(Br)",
            "OCF2CF2O", "SCF2CF2S", "NCF2CF2N",
        ]
        
        # ========== SELECTIVITY ENHANCING GROUPS ==========
        self.selectivity_enhancers = [
            # Bulky groups for selectivity pockets
            "C(CH3)3", "Si(CH3)3", "C(CF3)3", "C(Ph)3",
            "adamantyl", "cubyl", "dodecahedryl",
            "C1CC2CC3CC1CC(C2)C3",
            
            # H-bond network disruptors/enhancers 
            "CONH2", "CON(H)CH3", "CON(CH3)2", "CSNH2", "CSN(H)CH3",
            "SO2NH2", "SO2N(H)CH3", "PO3H2", "PO3(H)CH3", "AsO3H2",
            "c1nnn[nH]1", "c1nnc[nH]1", "c1noc[nH]1", "c1nsc[nH]1",
            
            # Entropy penalty reducers
            "c1ccc2ccccc2c1", "c1ccc2cc3ccccc3cc2c1",
            "C12CCCCC1CCCC2", "C1CC2CCCC3CCCC(C1)C23",
            "c1cc2nccnc2cc1", "c1cc2ncncc2cc1",
        ]
        logger.debug("ExternalMolecularGenerator initialized with enhanced EGFR TKI diversity.")

    def generate_molecules(self, n_molecules: int = 100,
                             seed_smiles: Optional[str] = None) -> List[str]:
        """
        Generates novel molecules using an Osimertinib-based strategy.

        Args:
            n_molecules (int): The number of molecules to generate.
            seed_smiles (Optional[str]): An optional SMILES string to use as a starting point.
                                         If None, Osimertinib's full SMILES is used.

        Returns:
            List[str]: A list of generated and validated unique SMILES strings.
        """
        generated_smiles = []
        
        logger.info(f"Generating {n_molecules} Osimertinib-like analogs focusing on EGFR mutations.")
        
        base_mol_smiles = seed_smiles if seed_smiles else self.osimertinib_full_smiles
        osimertinib_mol = Chem.MolFromSmiles(base_mol_smiles)
        
        if osimertinib_mol is None:
            logger.error(f"Could not parse base SMILES '{base_mol_smiles}'. Cannot generate analogs.")
            return []

        for i in range(n_molecules):
            current_smiles = Chem.MolToSmiles(osimertinib_mol) # Start each modification from original Osimertinib
            try:
                # Randomly choose a modification strategy
                modification_strategy = np.random.choice(
                    ["core_exchange", "side_chain_mod", "warhead_mod", "add_linker"],
                    p=[0.1, 0.5, 0.2, 0.2]
                )

                if modification_strategy == "core_exchange":
                    current_smiles = self._modify_core(current_smiles)
                elif modification_strategy == "side_chain_mod":
                    # Note: _mutate_osimertinib_side_chains primarily uses internal dictionary
                    current_smiles = self._mutate_osimertinib_side_chains(current_smiles, self.generation_2_modifiers)
                elif modification_strategy == "warhead_mod":
                    current_smiles = self._modify_warhead(current_smiles)
                elif modification_strategy == "add_linker":
                    current_smiles = self._add_new_linkers_or_motifs(current_smiles, self.generation_3_linkers)
                
                mol_candidate = Chem.MolFromSmiles(current_smiles)
                # Ensure sanitization for validity
                if mol_candidate:
                    try:
                        Chem.AllChem.SanitizeMol(mol_candidate, catchErrors=True)
                        generated_smiles.append(Chem.MolToSmiles(mol_candidate))
                    except Exception as e:
                        logger.warning(f"Generated SMILES '{current_smiles}' failed sanitization: {e}. Falling back to original Osimertinib for analog {i}.")
                        generated_smiles.append(Chem.MolToSmiles(osimertinib_mol))
                else:
                    logger.warning(f"Generated SMILES '{current_smiles}' is invalid. Falling back to original Osimertinib for analog {i}.")
                    generated_smiles.append(Chem.MolToSmiles(osimertinib_mol))

            except Exception as e:
                logger.warning(f"Error generating Osimertinib analog {i}: {e}. Falling back to original Osimertinib.", exc_info=False)
                generated_smiles.append(Chem.MolToSmiles(osimertinib_mol))

        # Validate and deduplicate
        valid_candidates = []
        seen_smiles = set()
        for smiles in generated_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    # Use a more robust sanitization flag for canonicalization
                    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    canonical_smiles = Chem.MolToSmiles(mol)
                    if canonical_smiles not in seen_smiles:
                        valid_candidates.append(canonical_smiles)
                        seen_smiles.add(canonical_smiles)
                except Exception as e:
                    logger.debug(f"Failed to sanitize or canonicalize SMILES '{smiles}': {e}")
                    pass # Skip invalid SMILES
        
        logger.info(f"Generated and validated {len(valid_candidates)} unique SMILES strings.")
        logger.debug("Molecules generated and deduplicated.")
        return valid_candidates

    def _modify_core(self, current_smiles: str) -> str:
        """
        Replace the pyrimidine core with an alternative core from self.core_aromaticities.

        Args:
            current_smiles (str): The SMILES string of the molecule to modify.

        Returns:
            str: The modified SMILES string, or the original if modification fails.
        """
        chosen_core_name = np.random.choice(list(self.core_aromaticities.keys()))
        new_core_smiles_fragment = self.core_aromaticities[chosen_core_name]
        
        # Identify the pyrimidine core to replace. This might need more sophisticated RDKit pattern matching
        # for real-world robustness if the pyrimidine isn't always exactly "c1nccc(n1)".
        if "c1nccc(n1)" in current_smiles:
            temp_smiles = current_smiles.replace("c1nccc(n1)", new_core_smiles_fragment, 1) # Replace only first occurrence
            mol_candidate = Chem.MolFromSmiles(temp_smiles)
            if mol_candidate:
                try:
                    Chem.AllChem.SanitizeMol(mol_candidate, catchErrors=True)
                    logger.debug(f"Core modified successfully with {chosen_core_name}.")
                    return Chem.MolToSmiles(mol_candidate)
                except Exception as e:
                    logger.warning(f"Core modification '{temp_smiles}' failed sanitization: {e}. Returning original SMILES.")
                    return current_smiles # Fallback
            else:
                logger.warning(f"Core modification '{temp_smiles}' resulted in invalid SMILES. Returning original SMILES.")
                return current_smiles
        
        logger.debug("Core modification not applied or original core not found, returning original SMILES.")
        return current_smiles

    def _modify_warhead(self, base_smiles: str) -> str:
        """
        Modify the acrylamide warhead with alternative covalent warheads.

        Args:
            base_smiles (str): The SMILES string of the molecule to modify.

        Returns:
            str: The modified SMILES string, or the original if modification fails.
        """
        # It's better to use fragments from self.osimertinib_fragments
        # For simplicity, keeping the direct mapping for now.
        warhead_replacements = {
            self.osimertinib_fragments["acrylamide_group"]: [
                self.osimertinib_fragments["propargyl_amide"], 
                self.osimertinib_fragments["vinyl_ketone"], 
                self.osimertinib_fragments["chloroacetamide"]
            ],
            # Add other warheads from self.osimertinib_fragments if needed
        }
        
        current_smiles = base_smiles
        modified_flag = False
        for original, alternatives in warhead_replacements.items():
            if original in current_smiles:
                chosen_replacement = np.random.choice(alternatives)
                temp_smiles = current_smiles.replace(original, chosen_replacement, 1) # Replace only first occurrence
                mol_temp = Chem.MolFromSmiles(temp_smiles)
                if mol_temp:
                    try:
                        Chem.AllChem.SanitizeMol(mol_temp, catchErrors=True)
                        logger.debug(f"Warhead modified successfully from {original} to {chosen_replacement}.")
                        return Chem.MolToSmiles(mol_temp)
                    except Exception as e:
                        logger.warning(f"Warhead modification '{temp_smiles}' failed sanitization: {e}. Skipping this modification.")
                        continue # Try next possible replacement or fall through
                else:
                    logger.warning(f"Warhead modification '{temp_smiles}' resulted in invalid SMILES. Skipping this modification.")
                    continue
        
        logger.debug("Warhead modification not applied, returning original SMILES.")
        return current_smiles

    def _mutate_osimertinib_side_chains(self, base_smiles: str, modifiers: List[str]) -> str:
        """
        Apply minor modifications to side chains of an Osimertinib-like molecule.
        This method primarily targets specific Osimertinib fragments defined internally.
        The 'modifiers' argument is currently not directly used for simple string replacement.

        Args:
            base_smiles (str): The SMILES string of the molecule to modify.
            modifiers (List[str]): A list of modifier SMILES fragments (currently not directly used
                                   for the pre-defined Osimertinib modifications).

        Returns:
            str: The modified SMILES string, or the original if modification fails.
        """
        current_smiles = base_smiles
        
        # Specific Osimertinib-based modifications using fragments from self.osimertinib_fragments
        osimertinib_modifications_map = {
            self.osimertinib_fragments["indole_substituent"]: [
                "c3cn(CC)c4ccccc34", "c3cn(CCF)c4ccccc34", "c3cnc4ccccc34",
                self.osimertinib_fragments["benzothiophene_substituent"], # Example from fragments
                self.osimertinib_fragments["quinoline_substituent"]       # Example from fragments
            ],
            self.osimertinib_fragments["dimethylamino_chain"]: [
                "N(CC)CCN(C)C", "NCCN(C)C", "N(C)CCCN(C)C",
                self.osimertinib_fragments["morpholine_chain"],           # Example from fragments
                self.osimertinib_fragments["piperazine_chain"]            # Example from fragments
            ],
            # Assuming 'COc1cc' is part of 'methoxy_aniline'
            self.osimertinib_fragments["methoxy_aniline"].split('N')[0]: [ # Targeting the methoxy group
                "CCOc1cc", "CFc1cc", "Oc1cc", 
                self.osimertinib_fragments["difluoro_methoxy_aniline"].split('N')[0] # Example
            ],
        }
        
        modified_flag = False
        for original, alternatives in osimertinib_modifications_map.items():
            if original in current_smiles and np.random.random() < 0.4: # 40% chance to apply
                chosen_replacement = np.random.choice(alternatives)
                temp_smiles = current_smiles.replace(original, chosen_replacement, 1)
                mol_temp = Chem.MolFromSmiles(temp_smiles)
                if mol_temp:
                    try:
                        Chem.AllChem.SanitizeMol(mol_temp, catchErrors=True)
                        current_smiles = Chem.MolToSmiles(mol_temp)
                        modified_flag = True
                        logger.debug(f"Side chain modified from {original} to {chosen_replacement}.")
                        break # Apply only one side chain modification per molecule for simplicity
                    except Exception as e:
                        logger.warning(f"Side chain modification '{temp_smiles}' failed sanitization: {e}. Skipping this modification.")
                        continue # Try next alternative or original
                else:
                    logger.warning(f"Side chain modification '{temp_smiles}' resulted in invalid SMILES. Skipping this modification.")
                    continue
        
        if modified_flag:
            return current_smiles
        else:
            logger.debug("Side chain mutation not applied, returning original SMILES.")
            return base_smiles # Return base_smiles if no successful modification occurred

    def _add_new_linkers_or_motifs(self, base_smiles: str, linkers: List[str]) -> str:
        """
        Attempts to add new linkers or motifs to the molecule.
        This is a simplistic implementation and might require more sophisticated
        graph-based approaches for chemically meaningful additions.

        Args:
            base_smiles (str): The SMILES string of the molecule to modify.
            linkers (List[str]): A list of SMILES fragments representing linkers.

        Returns:
            str: The modified SMILES string, or the original if modification fails.
        """
        mol = Chem.MolFromSmiles(base_smiles)
        if mol is None:
            logger.warning(f"Invalid base SMILES '{base_smiles}' for linker addition. Returning original.")
            return base_smiles

        chosen_linker = np.random.choice(linkers)
        current_smiles = base_smiles

        # This section needs more robust logic for attachment points and chemical validity.
        # Directly appending SMILES strings is often problematic for complex structures.
        eligible_atoms_for_extension = [atom for atom in mol.GetAtoms() 
                                        if atom.GetSymbol() in ['C', 'N', 'O', 'S'] and atom.GetTotalNumHs() >= 1]
        
        if eligible_atoms_for_extension and np.random.random() < 0.5: # 50% chance to attempt linker addition
            try:
                rwmol = Chem.RWMol(mol)
                target_atom = np.random.choice(eligible_atoms_for_extension)
                target_atom_idx = target_atom.GetIdx()
                
                # A very basic attempt: append a simple carbon chain or a common linker if chemically plausible
                # For complex linkers, you would need SMARTS-based attachment or bond cleavage/formation.
                if chosen_linker in ["CH2CH2", "C", "N", "O"]: # Limiting to simple atoms/chains for direct appending
                    # Create a molecule from the linker fragment
                    linker_mol = Chem.MolFromSmiles(chosen_linker)
                    if linker_mol:
                        # Find a suitable attachment point on the linker (e.g., first atom)
                        linker_attachment_idx = 0 
                        
                        # Merge the molecules and add a bond
                        combined_mol = Chem.CombineMols(rwmol, linker_mol)
                        rw_combined_mol = Chem.RWMol(combined_mol)
                        
                        # Original atom index in rw_combined_mol (after original molecule's atoms)
                        # Add bond between target_atom_idx (from original mol) and linker_attachment_idx (from linker_mol)
                        # The linker's atom indices will be shifted by the number of atoms in the original molecule
                        rw_combined_mol.AddBond(target_atom_idx, mol.GetNumAtoms() + linker_attachment_idx, Chem.BondType.SINGLE)
                        
                        # Attempt sanitization
                        try:
                            Chem.AllChem.SanitizeMol(rw_combined_mol)
                            logger.debug(f"New linker '{chosen_linker}' added successfully.")
                            return Chem.MolToSmiles(rw_combined_mol)
                        except Exception as e:
                            logger.warning(f"Combined molecule with linker '{chosen_linker}' failed sanitization: {e}. Returning original.")
                            return base_smiles
                    else:
                        logger.warning(f"Could not parse chosen linker SMILES '{chosen_linker}'. Skipping.")
                else:
                    logger.debug(f"Chosen linker '{chosen_linker}' is too complex for simple appending. Skipping.")
            except Exception as e:
                logger.warning(f"Error attempting to add linker: {e}. Returning original SMILES.")

        logger.debug("New linker/motif not added, returning original SMILES.")
        return current_smiles

    
class MolecularFilter:
    """Advanced molecular filtering system for EGFR-targeted compounds"""
    
    def __init__(self):
        self.mutation_db = EGFRMutationDatabase()
        print("11: MolecularFilter initialized.") # Checkpoint 11
        
    def calculate_drug_likeness(self, smiles: str) -> Dict[str, float]:
        """Calculate drug-likeness properties"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.error(f"Could not parse SMILES for property calculation: {smiles}")
                return {}
            
            properties = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'tpsa': Descriptors.TPSA(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
                'heavy_atoms': Descriptors.HeavyAtomCount(mol),
                'formal_charge': Chem.rdmolops.GetFormalCharge(mol),
                'qed': self._calculate_qed(mol)
            }
            print("12: Drug-likeness properties calculated.") # Checkpoint 12
            return properties
            
        except Exception as e:
            logger.error(f"Error calculating properties for {smiles}: {e}")
            return {}
    
    def _calculate_qed(self, mol) -> float:
        """Calculate Quantitative Estimate of Drug-likeness"""
        try:
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            psa = Descriptors.TPSA(mol)
            
            # Simplified QED scoring components
            mw_score = 1 - (mw - 350)**2 / 150000 if mw < 500 else 0
            logp_score = 1 - (logp - 2.5)**2 / 12.25 if -2 < logp < 5 else 0
            hbd_score = 1 - hbd/10 if hbd <= 5 else 0
            hba_score = 1 - hba/15 if hba <= 10 else 0
            psa_score = 1 - (psa - 90)**2 / 8100 if psa < 140 else 0
            
            # Ensure scores are not negative
            mw_score = max(0, mw_score)
            logp_score = max(0, logp_score)
            hbd_score = max(0, hbd_score)
            hba_score = max(0, hba_score)
            psa_score = max(0, psa_score)

            # Geometric mean for QED
            if all([mw_score, logp_score, hbd_score, hba_score, psa_score]):
                qed = (mw_score * logp_score * hbd_score * hba_score * psa_score)**(1/5)
            else:
                qed = 0.0
            
            print("13: QED calculated.") # Checkpoint 13
            return max(0.0, min(1.0, qed))
            
        except Exception as e:
            logger.error(f"Error calculating QED: {e}")
            return 0.0
    
    def lipinski_filter(self, properties: Dict[str, float]) -> bool:
        """Lipinski Rule of Five filter"""
        if not properties:
            return False
            
        result = (
            properties.get('molecular_weight', 501) <= 500 and
            properties.get('logp', 6) <= 5 and
            properties.get('hbd', 6) <= 5 and
            properties.get('hba', 11) <= 10
        )
        print(f"14: Lipinski filter applied. Pass: {result}") # Checkpoint 14
        return result
    
    def veber_filter(self, properties: Dict[str, float]) -> bool:
        """Veber filter for oral bioavailability"""
        if not properties:
            return False
            
        result = (
            properties.get('rotatable_bonds', 11) <= 10 and
            properties.get('tpsa', 141) <= 140
        )
        print(f"15: Veber filter applied. Pass: {result}") # Checkpoint 15
        return result
    
    def egfr_specific_filter(self, smiles: str, properties: Dict[str, float]) -> Dict[str, bool]: # Corrected type hint
        """EGFR-specific structural filters"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Could not parse SMILES for EGFR specific filter: {smiles}")
            return {'kinase_privileged': False, 'hinge_binder': False, 'gatekeeper_compatible': False}
        
        # Check for kinase-privileged structures
        kinase_scaffolds = [
            'c1cnc2c(c1)ncnc2',  # Quinazoline
            'c1cnc2c(c1)cccn2',  # Quinoline  
            'c1cc2c(cc1)cnc(n2)', # Pyrazolo-pyrimidine
        ]
        
        kinase_privileged = False
        for scaffold_smarts in kinase_scaffolds:
            try:
                scaffold_mol = Chem.MolFromSmarts(scaffold_smarts)
                if scaffold_mol and mol.HasSubstructMatch(scaffold_mol):
                    kinase_privileged = True
                    break
            except Exception as e:
                logger.debug(f"Error matching kinase scaffold '{scaffold_smarts}': {e}")
                continue
        
        # Check for hinge-binding motifs
        hinge_binders = [
            'c1[n;h]c(=O)nc1', # pyrimidine with NH in hinge region
            'c1nc(N)[nH]c1', # pyrimidine with amino group
            'c1ccc(-[#7]-c2ncncc2)', # anilino-pyrimidine like linker
            'c1ccc(-[#7]C(=O)C=C)', # acrylamide linker
        ]
        
        hinge_binder = False
        for pattern_smarts in hinge_binders:
            try:
                pattern_mol = Chem.MolFromSmarts(pattern_smarts)
                if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                    hinge_binder = True
                    break
            except Exception as e:
                logger.debug(f"Error matching hinge binder pattern '{pattern_smarts}': {e}")
                continue
        
        # Check T790M gatekeeper compatibility (often related to size and flexibility)
        gatekeeper_compatible = properties.get('molecular_weight', 999) < 600 and \
                                properties.get('rotatable_bonds', 15) < 12
        
        result = {
            'kinase_privileged': kinase_privileged,
            'hinge_binder': hinge_binder,  
            'gatekeeper_compatible': gatekeeper_compatible
        }
        print(f"16: EGFR specific filters applied: {result}") # Checkpoint 16
        return result
    
    def mutation_specific_scoring(self, smiles: str, target_mutations: List[str]) -> Dict[str, float]:
        """Score molecules based on their likelihood to overcome specific mutations"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Could not parse SMILES for mutation specific scoring: {smiles}")
            return {}
        
        scores = {}
        mol_wt = Descriptors.MolWt(mol)
        num_hbd = Descriptors.NumHDonors(mol)
        num_hba = Descriptors.NumHAcceptors(mol)
        num_rotb = Descriptors.NumRotatableBonds(mol)
        
        for mutation in target_mutations:
            mutation_info = self.mutation_db.get_mutation_info(mutation)
            if not mutation_info:
                logger.warning(f"Mutation info not found for {mutation}. Skipping scoring for this mutation.")
                continue                
            score = 0.0       

            if mutation == 'T790M':
                # T790M typically benefits from smaller size and non-covalent strategies or specific covalent binders
                score += max(0, (550 - mol_wt) / 550) * 0.3 # Prefer lower MW
                if not self._has_electrophilic_groups(mol): # T790M can develop resistance to irreversible inhibitors
                    score += 0.2  # Slight preference for non-covalent or reversible
                # Consider specific structural features for T790M, e.g., certain hinge binders
                if self.egfr_specific_filter(smiles, {'molecular_weight': mol_wt, 'rotatable_bonds': num_rotb}).get('gatekeeper_compatible', False):
                    score += 0.3 # Importance of gatekeeper compatibility
                
            elif mutation == 'C797S':
                # C797S renders covalent inhibitors ineffective. Allosteric or reversible binding is key.
                if not self._has_electrophilic_groups(mol):
                    score += 0.4 # Strong preference for non-covalent
                if self._has_allosteric_features(mol):
                    score += 0.3 # Reward potential allosteric binding
                # H-bond donors/acceptors can help with reversible binding
                score += min(1.0, (num_hbd + num_hba) / 10) * 0.2
                    
            elif mutation == 'L858R':
                # L858R increases kinase activity; often still sensitive to first/second gen TKIs.
                # Strong H-bonding with active site, balanced size.
                hb_score = min(1.0, (num_hbd + num_hba) / 8)
                score += hb_score * 0.3 # Good H-bonding network
                # Prefer molecules within a certain MW range, not too large
                score += max(0, (500 - abs(mol_wt - 450)) / 500) * 0.2 # Optimal around 450 Da
                if self.egfr_specific_filter(smiles, {}).get('hinge_binder', False):
                     score += 0.2 # Hinge binders are generally good for L858R
            elif mutation == 'exon19del':
                # Exon 19 deletions lead to altered ATP pocket. Flexibility can be important.
                flexibility = min(1.0, num_rotb / 10)
                score += flexibility * 0.2 # Some flexibility can adapt to altered pocket
                score += max(0, (480 - mol_wt) / 480) * 0.2 # Prefer slightly smaller/optimal size

            scores[mutation] = min(1.0, score) # Cap score at 1.0
        
        print("17: Mutation specific scores calculated.") # Checkpoint 17
        return scores
    
    def _has_electrophilic_groups(self, mol) -> bool:
        """Check for electrophilic groups that can form covalent bonds (e.g., Michael acceptors)"""
        electrophilic_patterns = [
            'C=CC(=O)N',     # Acrylamide (Osimertinib warhead)
            'C=CS(=O)(=O)N', # Sulfonyl acrylamide
            'C(=O)CCl',      # Alpha-chloro ketone/amide
            'C=C[C,S](=O)[N,O]', # General Michael acceptor
            '[#6]#[#6]C(=O)', # Propargyl amide
        ]
        
        for pattern in electrophilic_patterns:
            try:
                smart = Chem.MolFromSmarts(pattern)
                if smart and mol.HasSubstructMatch(smart):
                    return True
            except Exception as e:
                logger.debug(f"Error matching electrophilic pattern '{pattern}': {e}")
                continue
        return False
    
    def _has_allosteric_features(self, mol) -> bool:
        """Check for features that might enable allosteric binding"""
        mol_wt = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        aromatic_rings = Descriptors.NumAromaticRings(mol)

        # Allosteric inhibitors often have distinct properties: larger, more lipophilic, less flexible
        # These are general indicators and might need refinement based on specific allosteric sites.
        is_allosteric_candidate = (
            mol_wt > 450 and      # Tend to be larger
            logp > 3.0 and        # More lipophilic to access deeper pockets
            rot_bonds < 7 and     # Less flexible
            aromatic_rings >= 3   # Often have larger hydrophobic scaffolds
        )
        return is_allosteric_candidate

class EGFRMoleculeGenerator:
    """Main class that orchestrates the entire molecule generation and filtering pipeline"""
    
    def __init__(self, generator_type: str = "osimertinib_analogs"):
        self.molecular_generator = ExternalMolecularGenerator(generator_type=generator_type)
        self.molecular_filter = MolecularFilter()
        self.mutation_db = EGFRMutationDatabase()
        
        logger.info(f"EGFR Molecule Generator initialized with {generator_type} generator.")
        print("18: EGFRMoleculeGenerator class initialized.") # Checkpoint 18
    
    def generate_candidate_molecules(self, n_molecules: int = 1000,
                                     target_mutations: Optional[List[str]] = None) -> List[str]:
        """Generate candidate molecules targeting specific EGFR mutations"""
        
        if target_mutations is None:
            target_mutations = ['T790M', 'L858R', 'C797S', 'exon19del']
        
        logger.info(f"Generating {n_molecules} candidate molecules for mutations: {', '.join(target_mutations)}")

        candidates = self.molecular_generator.generate_molecules(n_molecules)
        print("19: Candidate molecules generated.") # Checkpoint 19
        return candidates
    
    def filter_and_score_molecules(self, candidates: List[str],
                                   target_mutations: List[str]) -> pd.DataFrame:
        """Filter and score candidate molecules"""
        
        logger.info(f"Filtering and scoring {len(candidates)} molecules...")
        
        results = []
        
        for i, smiles in enumerate(candidates):
            if i % 100 == 0:
                logger.info(f"Processing molecule {i+1}/{len(candidates)}")
            
            properties = self.molecular_filter.calculate_drug_likeness(smiles)
            if not properties:
                logger.debug(f"Skipping {smiles} due to missing properties.")
                continue
            
            lipinski_pass = self.molecular_filter.lipinski_filter(properties)
            veber_pass = self.molecular_filter.veber_filter(properties)
            egfr_filters = self.molecular_filter.egfr_specific_filter(smiles, properties) # Pass properties here
            
            mutation_scores = self.molecular_filter.mutation_specific_scoring(smiles, target_mutations)
            
            overall_score = 0.0
            
            # Incorporate filter passes into overall score
            if lipinski_pass:
                overall_score += 0.2
            if veber_pass:
                overall_score += 0.1
            if egfr_filters.get('kinase_privileged', False):
                overall_score += 0.2
            if egfr_filters.get('hinge_binder', False):
                overall_score += 0.2
            if egfr_filters.get('gatekeeper_compatible', False):
                overall_score += 0.1
            
            overall_score += properties.get('qed', 0.0) * 0.2 # QED contribution
            
            if mutation_scores:
                # Average mutation scores to contribute to overall score
                avg_mutation_score = sum(mutation_scores.values()) / len(mutation_scores)
                overall_score += avg_mutation_score * 0.2

            results.append({
                'smiles': smiles,
                'molecular_weight': properties.get('molecular_weight'),
                'logp': properties.get('logp'),
                'hbd': properties.get('hbd'),
                'hba': properties.get('hba'),
                'tpsa': properties.get('tpsa'),
                'rotatable_bonds': properties.get('rotatable_bonds'),
                'aromatic_rings': properties.get('aromatic_rings'),
                'heavy_atoms': properties.get('heavy_atoms'),
                'formal_charge': properties.get('formal_charge'),
                'qed': properties.get('qed'),
                'lipinski_pass': lipinski_pass,
                'veber_pass': veber_pass,
                **egfr_filters,
                **{f'score_{k}': v for k, v in mutation_scores.items()},
                'overall_score': overall_score
            })
        
        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results = df_results.sort_values(by=['overall_score', 'qed'], ascending=[False, False]).reset_index(drop=True)
            logger.info(f"Filtered and scored {len(df_results)} molecules.")
        else:
            logger.warning("No molecules passed initial filtering. Returning empty DataFrame.")

        print("20: Molecules filtered and scored.") # Checkpoint 20
        return df_results
    
    def select_top_candidates(self, scored_df: pd.DataFrame,
                              n_top: int = 50) -> pd.DataFrame:
        """Select top candidate molecules based on comprehensive scoring"""
        
        if scored_df.empty:
            logger.warning("Scored DataFrame is empty, cannot select top candidates.")
            return pd.DataFrame()

        # Strict criteria
        strict_candidates = scored_df[
            (scored_df['lipinski_pass'] == True) &
            (scored_df['veber_pass'] == True) &
            (scored_df['kinase_privileged'] == True) &
            (scored_df['overall_score'] > 0.6)
        ]
        
        if len(strict_candidates) >= n_top:
            logger.info(f"Selecting {n_top} candidates from strict criteria.")
            print(f"21: Selected {n_top} top candidates based on strict criteria.") # Checkpoint 21
            return strict_candidates.head(n_top)
        else:
            logger.warning(f"Only {len(strict_candidates)} candidates meet strict criteria. Relaxing criteria.")
            # Relax criteria
            relaxed_candidates = scored_df[
                (scored_df['lipinski_pass'] == True) &
                (scored_df['overall_score'] > 0.4)
            ]
            if len(relaxed_candidates) >= n_top:
                logger.info(f"Selecting {n_top} candidates from relaxed criteria.")
                print(f"22: Selected {n_top} top candidates based on relaxed criteria.") # Checkpoint 22
                return relaxed_candidates.head(n_top)
            else:
                logger.warning(f"Still only {len(relaxed_candidates)} candidates meet relaxed criteria. Returning all available.")
                print(f"23: Returning all {len(relaxed_candidates)} available candidates.") # Checkpoint 23
                return relaxed_candidates
    
    def generate_report(self, top_candidates: pd.DataFrame,
                        target_mutations: List[str]) -> str:
        """Generate a comprehensive report of the results"""
        
        if top_candidates.empty:
            logger.info("No top candidates found to report.")
            return "\nEGFR Mutation-Targeted Molecule Generation Report\n===============================================\nNo top candidates found to report."

        report = f"""
EGFR Mutation-Targeted Molecule Generation Report
===============================================

Target Mutations: {', '.join(target_mutations)}
Number of Top Candidates: {len(top_candidates)}

Summary Statistics (Top Candidates):
----------------------------------
Average Molecular Weight: {top_candidates['molecular_weight'].mean():.2f} Da
Average LogP: {top_candidates['logp'].mean():.2f}
Average QED Score: {top_candidates['qed'].mean():.3f}
Average Overall Score: {top_candidates['overall_score'].mean():.3f}

Filter Pass Rates (Top Candidates):
---------------------------------
Lipinski Rule of 5: {(top_candidates['lipinski_pass'].sum() / len(top_candidates) * 100):.1f}%
Veber Filter: {(top_candidates['veber_pass'].sum() / len(top_candidates) * 100):.1f}%
Kinase Privileged: {(top_candidates['kinase_privileged'].sum() / len(top_candidates) * 100):.1f}%
Hinge Binder: {(top_candidates['hinge_binder'].sum() / len(top_candidates) * 100):.1f}%
Gatekeeper Compatible: {(top_candidates['gatekeeper_compatible'].sum() / len(top_candidates) * 100):.1f}%

Mutation-Specific Scores (Average for Top Candidates):
------------------------------------------------------
"""
        for mutation in target_mutations:
            score_col = f'score_{mutation}'
            if score_col in top_candidates.columns:
                report += f"  {mutation}: {top_candidates[score_col].mean():.3f}\n"
            else:
                report += f"  {mutation}: N/A (Score column not found)\n"

        report += "\nTop Candidates (SMILES, Overall Score, QED):\n"
        for _, row in top_candidates.iterrows():
            report += f"SMILES: {row['smiles']}, Overall Score: {row['overall_score']:.3f}, QED: {row['qed']:.3f}\n"

        print("24: Report generated.") # Checkpoint 24
        return report
    
    def save_results(self, top_candidates: pd.DataFrame, output_path: str) -> None:
        """Save the top candidates DataFrame to a CSV file"""
        if top_candidates.empty:
            logger.warning("No candidates to save. Skipping save operation.")
            return
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        top_candidates.to_csv(output_file, index=False)
        logger.info(f"Top candidates saved to {output_file}")
        print(f"25: Results saved to {output_file}") # Checkpoint 25
    
    def load_results(self, input_path: str) -> pd.DataFrame:
        """Load results from a CSV file"""
        input_file = Path(input_path)
        if not input_file.exists():
            logger.error(f"Input file {input_file} does not exist.")
            return pd.DataFrame()
        
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} records from {input_file}")
        print(f"26: Results loaded from {input_file}") # Checkpoint 26
        return df
    
    def save_mutation_db(self, db_path: str) -> None:
        """Save the EGFR mutation database to a file"""
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(db_file, 'wb') as f:
            pickle.dump(self.mutation_db, f)
        logger.info(f"EGFR mutation database saved to {db_file}")
        print(f"27: Mutation DB saved to {db_file}") # Checkpoint 27
    
    def load_mutation_db(self, db_path: str) -> None:
        """Load the EGFR mutation database from a file"""
        db_file = Path(db_path)
        if not db_file.exists():
            logger.error(f"Mutation database file {db_file} does not exist.")
            return
        
        with open(db_file, 'rb') as f:
            self.mutation_db = pickle.load(f)
        logger.info(f"EGFR mutation database loaded from {db_file}")
        print(f"28: Mutation DB loaded from {db_file}") # Checkpoint 28
    
    def run_pipeline(self, n_molecules: int = 1000,
                     target_mutations: Optional[List[str]] = None,
                     output_path: str = "top_candidates.csv") -> str:
        """Run the entire molecule generation and filtering pipeline"""
        if target_mutations is None:
            target_mutations = ['T790M', 'L858R', 'C797S', 'exon19del']
        
        logger.info("Starting EGFR-targeted molecule generation pipeline...")
        print("29: Pipeline started.") # Checkpoint 29
        
        candidates = self.generate_candidate_molecules(n_molecules, target_mutations)       
        scored_molecules_df = self.filter_and_score_molecules(candidates, target_mutations)
        top_molecules_df = self.select_top_candidates(scored_molecules_df)
        self.save_results(top_molecules_df, output_path)
        report_output = self.generate_report(top_molecules_df, target_mutations)
        
        print("30: Pipeline finished successfully.") # Checkpoint 30
        return report_output

# Main execution flow
if __name__ == "__main__":
    egfr_system = EGFRMoleculeGenerator(generator_type="osimertinib_analogs") 
    print("31: Main execution started, EGFRMoleculeGenerator instantiated.") # Checkpoint 31

    # Step 1: Define target mutations
    target_mutations_for_generation = ['T790M', 'C797S', 'L858R', 'exon19del']
    logger.info(f"Targeting mutations: {', '.join(target_mutations_for_generation)}")

    # Step 2: Generate candidate molecules
    num_molecules_to_generate = 1500
    candidate_smiles = egfr_system.generate_candidate_molecules(
        n_molecules=num_molecules_to_generate,
        target_mutations=target_mutations_for_generation
    )
    logger.info(f"Generated {len(candidate_smiles)} candidate SMILES strings.")

    # Step 3: Filter and score molecules
    scored_molecules_df = egfr_system.filter_and_score_molecules(
        candidate_smiles,
        target_mutations_for_generation
    )
    logger.info(f"Scored {len(scored_molecules_df)} molecules.")

    # Step 4: Select top candidates
    num_top_candidates = 10
    top_molecules_df = egfr_system.select_top_candidates(
        scored_molecules_df,
        n_top=num_top_candidates
    )
    logger.info(f"Selected {len(top_molecules_df)} top candidates.")

    # Step 5: Generate report and save results
    if not top_molecules_df.empty:
        report_output = egfr_system.generate_report(
            top_molecules_df,
            target_mutations_for_generation
        )
        print("\n" + "="*80)
        print("Final Report:")
        print("="*80)
        print(report_output)
        
        # Save the top candidates to a CSV file
        egfr_system.save_results(top_molecules_df, "egfr_top_candidates.csv")
    else:
        logger.warning("No top candidates found after filtering. Consider relaxing criteria or generating more molecules.")

    print("32: Main execution finished.") # Checkpoint 32