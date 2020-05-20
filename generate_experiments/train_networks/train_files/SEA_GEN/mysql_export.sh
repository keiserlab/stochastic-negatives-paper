#!/bin/bash

# Copyright (C) 2013-2015 SeaChange Pharmaceuticals, Michael Mysinger

echo "For chembl 20+"
# Starting with chembl15 there are now multi-protein targets. 

database="chembl_20"
assay_type="B"
MYSQL="mysql -u ecaceres -p -h mk-1-b $database"

# Use common where clause to get the same slice of the data
SQLWHERE="assays.assay_type='$assay_type' AND assays.confidence_score>=4 AND assays.relationship_type in ('D', 'H') AND td.species_group_flag=0 AND td.target_type in ('SINGLE PROTEIN', 'PROTEIN COMPLEX', 'PROTEIN FAMILY');"

# Use common join because, for example, activity data without
#   compound structure is not useful
SQLFROM="assays JOIN target_dictionary AS td ON td.tid=assays.tid JOIN activities AS act ON act.assay_id=assays.assay_id JOIN molecule_dictionary as md ON md.molregno=act.molregno JOIN compound_structures as cs ON cs.molregno = act.molregno JOIN source ON source.src_id=assays.src_id"

echo "    Exporting targets."
SELECT="SELECT DISTINCT td.chembl_id, td.pref_name, td.target_type FROM $SQLFROM WHERE $SQLWHERE;"
echo $SELECT
echo $SELECT | $MYSQL > targets.txt

echo "    Exporting smiles."
SELECT="SELECT DISTINCT md.chembl_id, cs.canonical_smiles FROM $SQLFROM WHERE $SQLWHERE;"
echo $SELECT
echo $SELECT | $MYSQL > smiles.txt

echo "    Exporting activities."
SELECT="SELECT DISTINCT td.chembl_id, md.chembl_id, act.standard_units, act.standard_value, act.standard_type, act.standard_relation, act.doc_id, source.src_id, assays.confidence_score, assays.curated_by FROM $SQLFROM WHERE $SQLWHERE;"
echo $SELECT
echo $SELECT | $MYSQL > activities.txt

# For single proteins, pull mapping from ChEMBL id to uniprot accession

echo "    Exporting single protein accession codes."
SELECT="SELECT DISTINCT td.chembl_id, cseq.accession FROM assays JOIN target_dictionary AS td ON td.tid=assays.tid JOIN target_components as tc ON td.tid=tc.tid JOIN component_sequences as cseq ON tc.component_id=cseq.component_id WHERE assay_type='$assay_type' AND assays.confidence_score>=4 AND assays.relationship_type in ('D', 'H') AND td.target_type='SINGLE PROTEIN';"
echo $SELECT
echo $SELECT | $MYSQL > accession.txt

