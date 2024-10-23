from omegaconf import DictConfig

from text_hpo_mapping.data_preprocessor import CustomDataPreprocessor
from text_pheno_extractor import TextPhenoExtractor
from utils import start_metamap_servers
import hydra
from time import sleep
import logging
import pandas as pd


@hydra.main(version_base=None, config_path="config", config_name="config")
def run_extraction(cfg: DictConfig):

    logging.info(f"Phenotype extraction version {cfg.version}")

    # Step 1: Load data
    data = pd.read_excel(cfg.clinical_data_path)
    sample_id_column = 'Sample Name'
    preprocessor = CustomDataPreprocessor(data, sample_id_column)

    # Step 2: Initialize the pipeline
    logging.info("Initializing pipeline")
    txt_pheno = TextPhenoExtractor(cfg, preprocessor)

    # Step 3: Extract texts (from input data to {patient_id: text} dictionary)
    txt_pheno.extract_patient_texts()

    # Step 4: Prepare patient data (text pre-processing)
    txt_pheno.prepare_patient_data()

    # Step 5: Start MetaMap servers if needed
    if cfg.metamap.start_servers:
        logging.info("Starting MetaMap servers")
        start_metamap_servers(cfg)
        sleep(60)  # Ensure servers are up before proceeding

    # Step 6: Process each patient
    for pt_id in data['Sample Name'].tolist():
        try:
            logging.info(f"Processing patient {pt_id}")

            # Step 6a: Extract CUIs for the current patient
            selected_report = {pt_id: txt_pheno.patient_reports_dict[pt_id]}
            if selected_report[pt_id] == "":
                logging.info(f"Patient {pt_id}'s report is empty. Skipping")
                continue

            txt_pheno.extract_all_patient_cuis(selected_report) # selected_report only has a single pt report

            # Step 6b: Map CUIs to HPOs for the current patient
            txt_pheno.map_cuis_to_hpos([pt_id])  # Pass a list with the single patient ID

            # Step 6c: Refine HPOs for the current patient
            txt_pheno.refine_hpo_terms([pt_id])  # Pass a list with the single patient ID

            # Step 6d: Create csv file with extracted phenotypes
            txt_pheno.generate_csv_file([pt_id])

            logging.info(f"Successfully processed patient {pt_id}")

        except Exception as e:
            logging.error(f"Error processing patient {pt_id}: {e}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_extraction()

