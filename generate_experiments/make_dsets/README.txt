1. python mysql_generate_raw_data.py -a -c 800 username CHEMBL_RAW_DATA.csv.gz

2. zcat CHEMBL_RAW_DATA.csv.gz | tail -n +2 | awk -F$'\t' '{{print $4 "\t" $8}}' | sort -u -t$'\t' -k1,1 | gzip > CHEMBLMID_UNIQ_SMILES.smi

3. Create INCHI key look up for all molecules pre-fingerprinting: ./get_inchi.sh
    a. Gives file: chembl20_MWmax800_smiles2inchi2mid.csv.gz

4. Run fingerprinting: ./fingerprint_smiles.sh
    a. Input: 
        inchi2smiles.csv.gz
    b. Output: 
        chembl20_MWmax800_fps.fp.gz

5. Write scrubbed data: write_scrubbed.ipynb
    a. Output: 
        full_chembl20_cutoff800_dm_scrubbed.csv.gz

6. Make hdf5 
    a. Run File: ./run_hdf5_maker.sh
    b. Output: 
        chembl20_MWmax800_scrubDM_minpos10_cutoff5.hdf5
        chembl20_MWmax800_scrubDM_minpos10_cutoff5_target_index.pkl

7. Make Time Split:
    a. make_TS.ipynb
    b. Output: 
        train_ts2012_chembl20_MWmax800_scrubDM_minpos10_cutoff5.hdf5
        ts2012_chembl20_MWmax800_scrubDM_minpos10_cutoff5_target_index.pkl
        val_ts2012_chembl20_MWmax800_scrubDM_minpos10_cutoff5.hdf5
        
8. Make scrambled dataset
    a. make_scrambled_dset.ipynb