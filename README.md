# DRN-CDR
A Cancer Drug Response Prediction Model using Multi-Omics and Drug Features

This paper proposes a regression method using Deep ResNet for CDR (DRN-CDR) prediction that multi-omics data such as gene expressions, mutation data, and methylation data along with the molecular structural information of drugs to predict the IC50 values of drugs. 

- Cell Line features (genomic mutation data, Gene expression data, methylation data) are downloaded from CCLE database.

- Drug graph features are downloaded from PubChem database.

- IC50 values of Drug-Cell line pairs are downloaded from GDSC database


**Sample Data**

	* mutation.csv - Contains genomic mutation data of 561 Cell-Lines
	* methylation.csv - Contains DNA methylation data of 561 Cell-Lines 
	* gene_expression.csv - Contains gene expression data of 561 Cell-Lines
	* Cell_annotation.txt - Text file containing Cell-Line Ids and corresponding TCGA Cancer Types
	* GDSC_IC50.csv - ContainsIC50 values of Drug - Cell line pairs 
	* drug_graph_feat - folder containing hickle files of drug feature (Feature matrix, adjacency list, degree list)

**Program code**

 * run_DRNCDR.py - Run the main file to implement the model.
 * model.py - Model for drug response prediction.
