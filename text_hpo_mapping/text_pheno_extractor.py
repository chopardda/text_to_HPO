import os

import pandas as pd
from markdown_it.rules_inline import entity

from data_preprocessor import DataPreprocessor
from dataio.utils import *
from pyhpo import Ontology
from spacy import displacy
from utils import *
# %%
from tqdm import tqdm

tqdm.pandas()

import json
import re

# class TextPhenoExtractor is a class that extracts phenotypes from text

# triggers_to_ignore = ['severe', 'images', 'no', 'yes', 'normal', 'slightly', 'low', 'high', 'at', 'carried', 'centre',
#                       'therapy', 'well', 'deteriorated', 'history', 'abnormal', 'urine', 'very well', 'mutation', 'fit',
#                       'fell', 'cr', 'possible', 'norm', 'deletion', 'mr', 'problem', 'worsening', 'condition', 'ma',
#                       'asymptomatic', 'discharge', 'result', 'metabolic laboratory', 'break', 'very rare', 'a small',
#                       'follow-up', 'activity', 'biochemical', 'response', 'pl', 'detected', 'confirmed', 'rule',
#                       'possibility', 'parameters', 'growth', 'very low', 'very poor', 'mm', 'slightly below', 'in',
#                       'bad', 'others', 'mthfr', 'lowered', 'turns', 'religious', 'symptoms', 'persistent', 'much',
#                       'reached', 'little', 'normal results', 'mol', 'severely', 'difficult', 'worsened', 'turn', 'warm',
#                       'able', 'ie', 'carnitine', 'creatinine', 'likely', 'parameters', 'partial response', 'negative',
#                       'family history', 'moderate', 'abnormalities', 'maybe', 'somewhat', 'better', 'unable', 'clear',
#                       'treatment', 'favourable', 'progression', 'difficulty', 'solid', 'resolved', 'low level', 'iv',
#                       'platelets', 'status', 'sex', 'improved', 'mutations', 'finding', 'completely', 'oral', 'mild',
#                       'vitamin b12', 'ph', 'week', 'soft', 'positive', 'one', 'a lot', 'suffers', 'moderately', 'found',
#                       'slight', 'very little', 'resulted', 'but', 'was', 'cause of death', 'died', 'results', 'chr',
#                       'autopsy results', 'signs', 'fed', 'decreased', 'related', 'probable', 'uncertain', 'but',
#                       'complete', 'condition', 'better', 'three', 'disorder', 'carry', 'biopsy', 'completely', 'always',
#                       'underlying', 'mass', 'conditions', 'disease', 'discontinued', 'discontinue', 'hardly',
#                       'hospitalized', 'absence', 'effect', 'find', 'abnormality']


class TextPhenoExtractor:
    def __init__(self, cfg, data_preprocessor: DataPreprocessor):
        """
        Initialize the TextPhenoExtractor with the provided configuration and data preprocessor.

        :param cfg: Configuration object.
        :param data_preprocessor: A DataPreprocessor instance to handle data preparation.
        """
        self.cfg = cfg
        self.data_preprocessor = data_preprocessor
        self.patient_reports_dict = None
        self.patient_pheno_dataframes = {}  # Store all DataFrames per patient here
        self.patient_pheno_vis_dicts = {}
        self.html_colors_dict = {}
        _ = Ontology()

        # self.clinical_data_df = load_clinical_data(cfg.clinical_data_path)
        # self.hpo_target_phenotypes_df = load_hpo_target_phenotypes(cfg.hpo_target_phenotypes_path)
        # # self.prep_clinical_data_df = self._prepare_clinical_data()
        # self.prep_hpo_target_df = self._prepare_hpo_target_phenotypes()
        # self.cuis_only_df = load_cuis_only_data(cfg.cuis_only_path)

        # self.merged_pos_patient_pheno_dict = {}  # filled by method merge_concepts_at_positions

    # # prepare hpo target phenotypes
    # # - flatten the first three columns
    # # - criteria for selecting text column: not empty and not nan
    # # - remove phenotype (drop row) if HPO is missing
    # def _prepare_hpo_target_phenotypes(self):
    #     names = []
    #     codes = []
    #     # flatten the first three col
    #     for i, row in self.hpo_target_phenotypes_df.iterrows():
    #         if not pd.isna(row['HPO']):
    #             name = row[:4].dropna().iloc[-1] if not row[:4].dropna().empty else None
    #             names.append(name.strip())
    #             codes.append(row['HPO'])
    #
    #     # create a df with two columns: name and code
    #     prep_target_df = pd.DataFrame({'target': names, 'HPO': codes})
    #
    #     # remove nan
    #     prep_target_df.dropna(inplace=True)
    #
    #     return prep_target_df

    # # prepare clinical data
    # # - selecting right most text column per each patient (translated or original)
    # # - criteria for selecting text column: not emspty and not nan
    # def _prepare_clinical_data(self):
    #     text = []
    #     for i, row in self.clinical_data_df.iterrows():
    #         if not (pd.isna(row['clinicalDataTranslated']) or str(row['clinicalDataTranslated']).strip() == ''):
    #             text.append(row['clinicalDataTranslated'])
    #         else:
    #             if not (pd.isna(row['clinicalData']) or str(row['clinicalData']).strip() == ''):
    #                 text.append(row['clinicalData'])
    #             else:
    #                 text.append('')
    #     self.prep_clinical_data_df = self.clinical_data_df.copy()
    #     self.prep_clinical_data_df['text'] = [s.replace('\n', '\t') for s in
    #                                           text]  # so that metamap keeps the position right
    #     self.prep_clinical_data_df = self.prep_clinical_data_df[["Sample Name", "text"]]
    #     return self.prep_clinical_data_df

    # # get original clinical data df
    # def get_clinical_data_df(self):
    #     return self.clinical_data_df
    #
    # # get prepared clinical data df
    # def get_prep_clinical_data_df(self):
    #     return self.prep_clinical_data_df
    #
    # # get prepared hpo target phenotypes df
    # def get_prep_hpo_target_df(self):
    #     return self.prep_hpo_target_df

    # create patient pheno dict
    # key -> patient id
    # content -> pandas df with extracted phenotypes and relevant info

    def extract_patient_texts(self):
        """
        Extract patient texts using the data preprocessor and store them in the patient_reports_dict attribute.
        """
        self.patient_reports_dict = self.data_preprocessor.extract_texts()

    def prepare_patient_data(self):
        """
        Prepare the patient data by processing the extracted texts and ensuring that
        there are no newline characters, so the positions are correctly tracked by MetaMap.
        """
        if self.patient_reports_dict is None:
            raise ValueError("Patient texts have not been extracted. Call 'extract_patient_texts' first.")

        # Replace newlines with tabs to ensure correct position tracking by MetaMap
        for sample_id, text in self.patient_reports_dict.items():
            if pd.notna(text):
                self.patient_reports_dict[sample_id] = text.replace('\n', '\t')
            else:
                # Handle missing texts by replacing NaN with an empty string.
                # TODO: decide if best option
                self.patient_reports_dict[sample_id] = ''

    def select_patients_to_process(self, n_patients=10, list_of_pts=None):
        """
        Select a subset of patients to process based on the given criteria.

        :param n_patients: Number of patients to process. Default is 10.
        :param list_of_pts: A list of specific patient IDs to process. If provided, this will override n_patients.
        :return: A dictionary of selected patient reports to process.
        """
        if self.patient_reports_dict is None:
            raise ValueError("Patient texts have not been extracted. Call 'extract_patient_texts' first.")

        if list_of_pts:
            selected_reports = {pt_id: self.patient_reports_dict[pt_id] for pt_id in list_of_pts if
                                pt_id in self.patient_reports_dict}
        else:
            # Filter out empty reports and select the first n_patients
            filtered_reports = {pt_id: text for pt_id, text in self.patient_reports_dict.items() if text != ''}
            # Select up to n_patients from the filtered reports
            selected_reports = dict(list(filtered_reports.items())[:min(n_patients, len(filtered_reports))])

        return selected_reports

    def extract_all_patient_cuis(self, selected_reports):
        """
        Extract CUIs for each patient report using MetaMap, assuming patient data has been prepared.
        """
        if self.patient_reports_dict is None:
            raise ValueError("Patient data has not been prepared. Call 'prepare_patient_data' first.")

        # Iterate over the prepared data and run MetaMap
        for pt_id, text in selected_reports.items():
            try:
                print(f"Extracting CUIs with MetaMap for patient {pt_id}")
                df_cuis = self._extract_cuis_from_report(text)
                self.patient_pheno_dataframes[pt_id] = {'df_cuis': df_cuis}
                self.separate_pos_info_and_trigger_words(pt_id)
                self.visualize_and_save(pt_id, 'df_cuis', step_name="cuis_extraction")
                # self.separate_pos_info_and_trigger_words(pt_id)
                # self.patient_cuis_dict[pt_id]['hpo_codes'] = self.map_cuis_to_hpo_codes(self.patient_cuis_dict[pt_id])

            except Exception as e:
                print(f"Problem with MetaMap for patient {pt_id}: {e}")
                continue

    def map_cuis_to_hpo_codes(self, cuis_df):
        """
        Map CUIs in a patient's DataFrame to HPO codes using the refactored mapping function.

        :param cuis_df: DataFrame containing CUIs for a patient.
        :return: List of HPO codes.
        """
        return map_cuis_to_hpo_codes(cuis_df, self.cfg)

    def map_cuis_to_hpos(self, patient_ids):
        """
        Map CUIs to HPO terms and update DataFrame.

        :param cuis_df: DataFrame containing CUIs for a patient.
        :return: List of HPO codes.
        """
        for pt_id in patient_ids:
            try:
                print(f"Mapping CUis to HPOs for patient {pt_id}")

                df_cuis = self.patient_pheno_dataframes[pt_id]['df_cuis']
                df_cuis_hpos = df_cuis.copy()

                # Map CUIs to HPO codes
                df_cuis_hpos['hpo_codes'] = self.map_cuis_to_hpo_codes(df_cuis_hpos)

                # Clean up and refine mapping
                df_cuis_hpos_cleaned = self._clean_and_refine_hpo_mapping(df_cuis_hpos)

                # Store the intermediate DF (after cleaning)
                self.patient_pheno_dataframes[pt_id]['df_cuis_hpos'] = df_cuis_hpos_cleaned
                print(self.patient_pheno_dataframes[pt_id]['df_cuis_hpos'])

                # Visualize the cleaned data before refinement
                self.visualize_and_save(pt_id, 'df_cuis_hpos', step_name="cuis_to_hpo_mapping")

                # self.refine_hpo_terms(pt_id)
            except Exception as e:
                print(f"Problem mapping CUIs to HPO for patient {pt_id}: {e}")
                continue

    def _clean_and_refine_hpo_mapping(self, df_cuis_hpos):
        """
        Clean up and refine the HPO mapping by:
        1. Removing rows where 'hpo_codes' is None.
        2. Flattening rows with multiple HPO codes into separate rows.
        3. Keeping relevant columns: 'pos_info', 'trigger_word', 'cui', 'hpo_code', 'hpo_name'.
        4. Consolidating rows with the same values except 'cui' by concatenating 'cui' values.
        5. If a 'trigger_word' at a given 'pos_info' is mapped to multiple 'hpo_code' and one 'hpo_code' has
        a name that perfectly matches the trigger_word (ignoring capitalization), keep only that row.

        :param df_cuis_hpos: DataFrame with CUIs and mapped HPO codes.
        :return: Cleaned and refined DataFrame.
        """
        refined_rows = []

        for _, row in df_cuis_hpos.iterrows():
            hpo_codes = row['hpo_codes']

            # Skip rows with None in hpo_codes
            if hpo_codes is None:
                continue

            # Iterate over the list of HPO codes and create a new row for each
            for hpo_entry in hpo_codes:
                refined_row = {
                    'pos_info': row['pos_info'],
                    'trigger_word': row['trigger_word'],
                    'cui': row['cui'],
                    'hpo_code': hpo_entry['hpo'],  # Extract 'hpo' from the dict
                    'hpo_name': hpo_entry['name'],  # Extract 'name' from the dict
                }
                refined_rows.append(refined_row)

        # Convert the refined rows back to a DataFrame
        df_cuis_hpos_cleaned = pd.DataFrame(refined_rows,
                                            columns=['pos_info', 'trigger_word', 'cui', 'hpo_code', 'hpo_name'])

        # Group CUIs and aggregate them in a list
        df_cuis_hpos_cleaned = df_cuis_hpos_cleaned.groupby(['pos_info', 'trigger_word', 'hpo_code', 'hpo_name'],
                                                            as_index=False).agg({
            'cui': lambda x: list(set(x))  # Concatenate CUIs into a list, removing duplicates
        })

        # Step 5: Keep only the row where 'hpo_name' matches 'trigger_word' exactly (ignoring case) if it exists
        def normalize_word(word):
            # Lowercase and remove trailing 's' or 'S' for comparison
            return word.lower().rstrip('s')

        def filter_rows(group):
            # Normalize both 'hpo_name' and 'trigger_word' by ignoring case and removing trailing 's'
            perfect_match = group[
                group['hpo_name'].apply(normalize_word) == group['trigger_word'].apply(normalize_word)]
            if not perfect_match.empty:
                return perfect_match
            return group

        df_cuis_hpos_cleaned = df_cuis_hpos_cleaned.groupby(['pos_info', 'trigger_word']).apply(
            filter_rows).reset_index(drop=True)

        return df_cuis_hpos_cleaned

    def refine_hpo_terms(self, patient_ids):
        """
        Apply refinement steps to the DataFrame.
        a) Remove HPO codes that are not children of 'Phenotypic Abnormality'.
        b) If trigger_word is composed of multiple words, remove individual words from the same position.
        c) If multiple HPOs match a single trigger word at some position, keep only the leaf-most one.

        :return: Refined DataFrames for each step.
        """
        phenotypic_abnormality = Ontology.get_hpo_object('HP:0000118')  # Phenotypic Abnormality

        for pt_id in patient_ids:
            try:
                df_cuis_hpos = self.patient_pheno_dataframes[pt_id]['df_cuis_hpos']
                df_hpo_refinement_1 = self._remove_non_abnormal_hpo(df_cuis_hpos, phenotypic_abnormality)
                df_hpo_refinement_2 = self._remove_individual_trigger_words(df_hpo_refinement_1)
                df_hpo_refinement_3 = self._keep_leaf_most_hpo(df_hpo_refinement_2)

                # Store the refined DataFrames at the end of the refinement steps
                self.patient_pheno_dataframes[pt_id]['df_hpo_refinements'] = df_hpo_refinement_3

                # Visualize and save each refinement step
                self.visualize_and_save(pt_id, 'df_hpo_refinements', step_name="hpo_refinement")

            except Exception as e:
                print(f"Problem refining HPO terms for patient {pt_id}: {e}")
                continue

    def generate_csv_file(self, patient_ids):
        for pt_id in patient_ids:
            try:
                output_dir = os.path.join(self.cfg.output_path, self.cfg.version, "table_output")
                os.makedirs(output_dir, exist_ok=True)
                csv_output_path = os.path.join(output_dir, f'{pt_id}.csv')
                print('writing file to', csv_output_path)
                self.patient_pheno_dataframes[pt_id]['df_hpo_refinements'].to_csv(csv_output_path, index=False)
            except Exception as e:
                print(f"Problem writing csv output for {pt_id}: {e}")
                continue

    def _remove_non_abnormal_hpo(self, df, phenotypic_abnormality):
        """
        Remove HPO codes that are not children of 'Phenotypic Abnormality'.
        """
        refined_rows = []
        for _, row in df.iterrows():
            try:
                hpo_term = Ontology.get_hpo_object(row['hpo_code'])
                if hpo_term and hpo_term.child_of(phenotypic_abnormality):
                    refined_rows.append(row)  # Keep only if it's a child of 'Phenotypic Abnormality'
            except Exception as e:
                print(f"Error processing HPO term {row['hpo_code']}: {e}")
                continue
        return pd.DataFrame(refined_rows, columns=df.columns)

    def _remove_individual_trigger_words(self, df):
        """
        If trigger_word consists of multiple words, remove individual words from the same position.
        """
        refined_rows = []
        for _, row in df.iterrows():
            if len(row['trigger_word'].split()) > 1:
                # If trigger_word is a phrase, remove individual words at the same position
                refined_rows.append(row)
            else:
                # Check if there are other rows at the same position with multi-word trigger
                multi_word_rows = df[
                    (df['pos_info'] == row['pos_info']) & (df['trigger_word'].str.split().str.len() > 1)]
                if multi_word_rows.empty:
                    refined_rows.append(row)
        return pd.DataFrame(refined_rows, columns=df.columns)

    def _keep_leaf_most_hpo(self, df):
        """
        If multiple HPOs match a single trigger word at the same position, keep only the leaf-most HPO.
        """

        def is_leaf_most(hpo_code, hpo_list):
            term = Ontology.get_hpo_object(hpo_code)
            if not term:
                return False
            return not any(Ontology.get_hpo_object(hpo).is_parent_of(term) for hpo in hpo_list if hpo != hpo_code)

        refined_rows = []
        grouped = df.groupby(['pos_info', 'trigger_word'])
        for _, group in grouped:
            hpo_codes = group['hpo_code'].tolist()
            leaf_most_hpos = [hpo for hpo in hpo_codes if is_leaf_most(hpo, hpo_codes)]
            refined_rows.extend(group[group['hpo_code'].isin(leaf_most_hpos)].to_dict(orient='records'))

        return pd.DataFrame(refined_rows, columns=df.columns)

    def visualize_and_save(self, pt_id, df_key, step_name):
        """
        Visualize the current state of the patient's data and save it as an HTML file.
        """
        df = self.patient_pheno_dataframes[pt_id][df_key]
        ent_label_col = 'cui' if df_key == 'df_cuis' else 'hpo_code'
        ent_name_col = 'preferred_name' if df_key == 'df_cuis' else 'hpo_name'
        visu_type = 'cuis' if df_key == 'df_cuis' else 'hpo'
        vis_dict, color_mapping = self._create_visualization_dict(pt_id, df, ent_label_col, ent_name_col, visu_type)
        self.patient_pheno_vis_dicts[pt_id] = vis_dict
        self.html_colors_dict.update(color_mapping)

        output_dir = f"{self.cfg.output_path}{self.cfg.version}/{step_name}"
        os.makedirs(output_dir, exist_ok=True)
        html_output_path = f"{output_dir}/{pt_id}_{step_name}.html"
        self._save_html_visualization(html_output_path, vis_dict)

    def _save_html_visualization(self, output_path, vis_dict):
        """
        Save the visualization dictionary as an HTML file using displacy.
        """
        extended_options = {"ents": list(self.html_colors_dict.keys()), "colors": self.html_colors_dict}
        html = displacy.render(vis_dict, style="ent", manual=True, page=True, options=extended_options)
        with open(output_path, "w") as f:
            f.write(html)



    def _extract_cuis_from_report(self, text):
        # extract CUIs from text with MetaMap (output result in pandas df)
        cuis = self._run_metamap_extraction(text)
        return cuis

    def _run_metamap_extraction(self, text):
        try:
            cuis = extract_cuis(note=text, cfg=self.cfg)
            return cuis
        except Exception as e:
            print(f"MetaMap extraction failed: {e}")
            return pd.DataFrame()

    def separate_pos_info_and_trigger_words(self, pt_id):
        """
        Process each patient CUIs and separate the 'pos_info' and 'trigger' fields into
        individual rows, creating a more granular DataFrame for analysis.
        """
        pt_df = self.patient_pheno_dataframes[pt_id]['df_cuis'].copy()

        new_rows = []
        for idx, row in pt_df.iterrows():
            pos_infos = row['pos_info'].split(';')
            triggers = custom_parse_list(row['trigger'])

            for pos_info, trigger in zip(pos_infos, triggers):
                if '[' in pos_info:
                    pos_info = pos_info.replace('[', '').replace(']', '')
                    sub_pos_infos = pos_info.split(',')
                    for sub_pos_info in sub_pos_infos:
                        trigger_word = extract_trigger_word(trigger)
                        new_row = row.to_dict()
                        new_row['pos_info'] = sub_pos_info
                        new_row['trigger'] = trigger
                        new_row['trigger_word'] = trigger_word
                        new_rows.append(new_row)
                else:
                    trigger_word = extract_trigger_word(trigger)
                    new_row = row.to_dict()
                    new_row['pos_info'] = pos_info
                    new_row['trigger'] = trigger
                    new_row['trigger_word'] = trigger_word
                    new_rows.append(new_row)

        self.patient_pheno_dataframes[pt_id]['df_cuis'] = pd.DataFrame(new_rows)

    def _create_visualization_dict(self, pt_id, df, entity_label_col, entity_name_col, vis_type):
        vis_dict = {
            'text': self.patient_reports_dict[pt_id],
            'title': pt_id,
            'ents': []
        }

        ents_df = pd.DataFrame(columns=['start', 'end', 'label', 'kb_id', 'semtype', 'color'])
        use_semtype_colors = self.cfg.visualization.use_semtype_colors

        for _, row in df.iterrows():
            positions = re.split(',', row['pos_info'])
            for position in positions:
                pos = position.split('/')
                start_char = int(pos[0]) - 1
                end_char = start_char + int(pos[1])

                # Determine the color based on the flag
                if use_semtype_colors and vis_type == 'cuis':
                    semtype = row['semtypes'].strip('[]')
                    color = self.cfg.visualization.colors.cui_semtype_colors.get(semtype,
                                                                                 self.cfg.visualization.colors.cui)
                elif vis_type == 'cuis':
                    semtype = ''
                    color = self.cfg.visualization.colors.cui
                else:
                    semtype = ''
                    color = self.cfg.visualization.colors.hpo

                ents_df.loc[len(ents_df.index)] = [start_char, end_char, row[entity_label_col], row[entity_name_col],
                                                   semtype, color]

        ents_df_filtered = ents_df#.groupby(['start', 'end'])
        ents_df_filtered['kb_id'] = ents_df_filtered['kb_id'].apply(join_list_to_string) # if there are multiple label names associated with one code

        vis_dict['ents'] = ents_df_filtered[['start', 'end', 'label', 'kb_id']].to_dict('records') # dataframe to dictionary {col_name: col_value}

        # Create a color mapping dictionary where each unique label is mapped to the specified color
        # unique_labels = ents_df_filtered['label'].unique()
        # color_mapping = {label: color for label in unique_labels}
        color_mapping = {label: ents_df_filtered.loc[ents_df_filtered['label'] == label, 'color'].iloc[0] for label in
                         ents_df_filtered['label'].unique()}

        return vis_dict, color_mapping

    # Function to join list elements into a string
    def join_list_to_string(lst):
        if isinstance(lst, list):
            return '; '.join(lst)
        return lst

    def get_patient_reports(self):
        """
        Return the prepared patient data.

        :return: A dictionary mapping 'sample_id' to 'text'.
        """
        return self.patient_reports_dict