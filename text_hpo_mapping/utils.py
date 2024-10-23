import json
import os
import re
import requests

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pymetamap import MetaMap


# single conversion from CUI to HPO
# input: cui
# output: {name,hpo} dict
# example: input: C0333913
# example: output: {'name': 'Albinism', 'hpo': 'HP:0001022'}
def convert_cui_to_hpo(cui, cfg):
    page = 0
    sabs = "HPO"
    version = '2023AB'

    # print('SEARCH CUI: ' + cui + '\n' + '\n')

    all_results = []

    try:
        while True:
            page += 1
            path = f'/search/{version}'
            query = {
                'apiKey': cfg.api_key,
                'string': cui,
                'sabs': sabs,
                'returnIdType': 'code',
                'pageNumber': page
            }
            output = requests.get(cfg.base_uri + path, params=query)
            output.encoding = 'utf-8'
            output_json = output.json()

            results = output_json.get('result', {}).get('results', [])
            if results:
                all_results.extend(results)
            if not results:
                break

    except Exception as e:
        print(f"Error fetching HPO for CUI {cui}: {e}")
        return [None]

    all_items = [{"cui": cui, "name": item['name'], "hpo": item['ui']} for item in all_results]
    return all_items if all_items else [None]


#     results = (([output_json['result']])[0])['results']
    #     if len(results) != 0:
    #         # print(cui, results)
    #         all_results.append(results)
    #
    #     if len(results) == 0:  # when there is no more result found, break
    #         if page == 1:
    #             # print('No results found for ' + cui + '\n')
    #             return [None]
    #         else:
    #             break
    #
    #     # for item in results:
    #     #     print('Name: ' + item['name'] + '\n' + 'UI: ' + item['ui'] + '\n' + 'Source Vocabulary: ' + item[
    #     #         'rootSource'] + '\n' + 'URI: ' + item['uri'] + '\n' + '\n')
    #
    # all_items = []
    # for results in all_results:
    #     for item in results:
    #         all_items.append({"cui": cui, "name": item['name'], "hpo": item['ui']})
    #
    # # print(all_items)
    # # print('***' + '\n' + '\n')
    # return all_items


#
# # Function to get unique HPO codes for a given CUI
# def get_unique_hpo_codes(cui):
#     codes = [hpo for (c, hpo) in cui_to_hpo_info.keys() if c == cui]
#     return codes if codes else None
#
#
# # Function to get unique HPO names for a given CUI
# def get_unique_hpo_names(cui):
#     codes = get_unique_hpo_codes(cui)
#     if codes is None:
#         return None
#     return [cui_to_hpo_info[(cui, hpo)] for hpo in codes]


# extract cuis from a single note
def extract_cuis(note, cfg):
    # assert note is a str
    assert isinstance(note, str), 'note must be a string'
    note = [note]
    metam = MetaMap.get_instance(cfg.metamap.base_dir + cfg.metamap.bin_dir)

    # Metamap semantic types in semantic group "disorders" and corresponding mapping
    # DISO | Disorders | T020 | Acquired Abnormality -> acab
    # DISO | Disorders | T190 | Anatomical Abnormality -> anab
    # DISO | Disorders | T049 | Cell or Molecular Dysfunction -> comd
    # DISO | Disorders | T019 | Congenital Abnormality -> cgab
    # DISO | Disorders | T047 | Disease or Syndrome -> dsyn
    # DISO | Disorders | T050 | Experimental Model of Disease
    # DISO | Disorders | T033 | Finding -> fndg
    # DISO | Disorders | T037 | Injury or Poisoning -> inpo
    # DISO | Disorders | T048 | Mental or Behavioral Dysfunction -> mobd
    # DISO | Disorders | T191 | Neoplastic Process -> neop
    # DISO | Disorders | T046 | Pathologic Function -> patf
    # DISO | Disorders | T184 | Sign or Symptom -> sosy

    cons, errs = metam.extract_concepts(note,
                                        word_sense_disambiguation=False,
                                        restrict_to_sts=['sosy', 'fndg', 'dsyn', 'mobd',
                                                         'acab', 'anab', 'comd', 'cgab', 'inpo',
                                                         'neop', 'patf'],
                                        composite_phrase=10,  # for memory issues
                                        unique_acronym_variants=True,
                                        # allow_overmatches=True,
                                        prune=30,
                                        no_nums=['fndg', 'dsyn', 'sosy'])
    # Choose keys from metamap that we are interested in
    keys_of_interest = ['preferred_name', 'cui', 'semtypes', 'trigger', 'pos_info']
    cols = [get_keys_from_mm(cc, keys_of_interest) for cc in cons]
    # Create a pandas dataframe from those keys
    results_df = pd.DataFrame(cols, columns=keys_of_interest)
    return results_df


# Function to filter rows within each group
def merge_phenos_if_child_found(group):
    # Check if there's any row in the group where 'hpo_codes' matches 'target_pheno_HPO'
    matching_rows = group[group['hpo_codes'] == group['target_phenotype_HPO']]

    # If there's a matching row, return only that row
    if not matching_rows.empty:
        return matching_rows
    # If there's no matching row, return the entire group
    return group


# Custom function to replace ',' with ';' and remove '[' and ']' if they are present
def replace_and_remove_brackets(value):
    if isinstance(value, str):
        if '[' in value or ']' in value:
            return value.replace(',', ';').replace('[', '').replace(']', '')
    return value


# Function to flatten the list and remove duplicates while preserving order
def remove_duplicates_preserve_order(lst):
    seen = set()
    return [x for x in lst if x is not None and not (x in seen or seen.add(x))]


def get_ancestors_from_hpo(hpo, cfg):
    assert isinstance(hpo, str), 'hpo must be a string'

    source = 'HPO'
    version = '2023AB'
    operation = 'ancestors'
    content_endpoint = "/rest/content/" + version + "/source/" + source + "/" + hpo + "/" + operation

    ancestors = []

    pageNumber = 0

    try:
        while True:
            pageNumber += 1
            query = {'apiKey': cfg.api_key, 'pageNumber': pageNumber}
            r = requests.get(cfg.base_uri + content_endpoint, params=query)
            r.encoding = 'utf-8'
            items = r.json()

            if r.status_code != 200:
                if pageNumber == 1:
                    print('No results found.' + '\n')
                    break
                else:
                    break

            # print("Results for page " + str(pageNumber) + "\n")

            for result in items["result"]:
                # print(result)
                if result["rootSource"] == 'HPO':
                    try:
                        ancestors.append((result['ui'], result['name']))  # TODO: APPEND THE NAME FROM PHENOTYPE LIST
                    except:
                        print('error in ancestors of ' + hpo, result)

    except Exception as except_error:
        print(except_error)

    return ancestors


def get_parents_from_hpo(hpo, cfg):
    assert isinstance(hpo, str), 'hpo must be a string'

    source = 'HPO'
    version = '2023AB'
    operation = 'parents'
    content_endpoint = "/rest/content/" + version + "/source/" + source + "/" + hpo + "/" + operation

    parents = []

    pageNumber = 0

    try:
        while True:
            pageNumber += 1
            query = {'apiKey': cfg.api_key, 'pageNumber': pageNumber}
            r = requests.get(cfg.base_uri + content_endpoint, params=query)
            r.encoding = 'utf-8'
            items = r.json()

            if r.status_code != 200:
                if pageNumber == 1:
                    print('No results found.' + '\n')
                    break
                else:
                    break

            # print("Results for page " + str(pageNumber) + "\n")

            for result in items["result"]:
                # print(result)
                if result["rootSource"] == 'HPO':
                    try:
                        # parents.append(result['ui'])
                        parents.append((result['ui'], result['name']))  # TODO: APPEND THE NAME FROM PHENOTYPE LIST
                    except:
                        print('error in ancestors of ' + hpo, result)

    except Exception as except_error:
        print(except_error)

    return parents


def get_keys_from_mm(concept, klist):
    conc_dict = concept._asdict()
    conc_list = [conc_dict.get(kk) for kk in klist]
    return (tuple(conc_list))


def start_metamap_servers(cfg):
    os.system(cfg.metamap.base_dir + cfg.metamap.pos_server_dir + ' start')  # Part of speech tagger
    os.system(cfg.metamap.base_dir + cfg.metamap.wsd_server_dir + ' start')  # Word sense disambiguation


# Function to filter rows in each group (remove rows from 'only_CUI' and 'HPO_not_in_target' if already 'HPO_found'
# with same start and end position)
def filter_rows(group):
    if any(group['category'] == 'HPO_found'):
        # Remove rows with specific categories if 'HPO_found' is present in the group
        return group[group['category'] != 'only_CUI'][group['category'] != 'HPO_not_in_target']
    return group


# Function to join list elements into a string
def join_list_to_string(lst):
    if isinstance(lst, list):
        return '; '.join(lst)
    return lst


# function to extract figure tags from html
def extract_figure(html):
    soup = BeautifulSoup(html, 'html.parser')
    figure = soup.find('figure')
    return figure


def build_html_from_figures(figures):
    # body settings copied from the displacy.render function
    html = '''<body style="font-size: 16px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica,
     Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; padding: 4rem 2rem; direction: ltr">'''

    for figure in figures:
        html += str(figure)
    html += "</body></html>"
    return html

def map_cuis_to_hpo_codes(cuis_df, cfg):
    """
    Map CUIs in a DataFrame to HPO codes.

    :param cuis_df: DataFrame containing CUIs.
    :param cfg: Configuration object containing API details.
    :return: List of mapped HPO codes.
    """
    hpo_codes = []
    for _, row in cuis_df.iterrows():
        try:
            hpo_codes_matched = convert_cui_to_hpo(cui=row['cui'], cfg=cfg)
            # print('>>>>>>', row['trigger_word'], ': ', row['cui'], ' (', row['preferred_name'], ') -> ', hpo_codes_matched)

            if hpo_codes_matched[0] is None:
                hpo_codes_matched = []

                # Generate n-grams from trigger words
                all_ngrams = generate_ngrams(row['trigger_word'], len(row['trigger_word'].split()))

                # Generate n-grams from preferred names
                preferred_name_ngrams = generate_ngrams(row['preferred_name'], len(row['preferred_name'].split()))
                all_ngrams.extend(preferred_name_ngrams)

                all_ngrams = list(set(all_ngrams))  # Remove duplicates

                # print("all_ngrams:", all_ngrams)
                for ngram in all_ngrams:
                    if ngram.lower() not in ['severe', 'mild', 'normal']:
                        temp_df = extract_cuis(ngram, cfg)
                        # print("temp_df", temp_df)
                        for _, row_jj in temp_df.iterrows():
                            ngram_matches = convert_cui_to_hpo(cui=row_jj['cui'], cfg=cfg)
                            # print("ngram_matches", ngram_matches)
                            hpo_codes_matched.extend(ngram_matches)

            hpo_codes_matched = [item for item in hpo_codes_matched if item]
            if not hpo_codes_matched:
                hpo_codes.append(None)
            else:
                unique_hpos = {json.dumps(d, sort_keys=True) for d in hpo_codes_matched}
                hpo_codes.append([json.loads(hpo) for hpo in unique_hpos])
        except Exception as e:
            print(f"Error mapping CUI to HPO: {e}")
            hpo_codes.append(None)

    return hpo_codes

def get_all_branches(hpo_tuple, cfg, current_branch=None):
    if current_branch is None:
        current_branch = []

    # Append the current hpo_tuple to the branch
    current_branch.append(hpo_tuple)

    # Get the parents of the current HPO code (as tuples of code and name)
    parents = get_parents_from_hpo(hpo_tuple[0], cfg)  # Use hpo_code from the tuple

    # If there are no parents, this is a root or an isolated node
    if not parents:
        return [current_branch]

    # List to store all branches
    branches = []

    # For each parent, get its branches and add them to the list
    for parent_tuple in parents:
        branches.extend(get_all_branches(parent_tuple, cfg, current_branch.copy()))

    return branches


def generate_ngrams(text, max_n):
    """
    Generate n-grams from the given text up to n.

    :param text: The input text.
    :param max_n: Maximum n for n-grams.
    :return: List of n-grams.
    """
    words = text.split()
    ngrams = [' '.join(words[i:i+n]) for n in range(1, max_n+1) for i in range(len(words) - n + 1)]
    return ngrams


def custom_parse_list(string_list):
    '''
    To transform column ['trigger'] which is a string representing a list (metamap output) to a list of strings
    '''
    # Remove square brackets and split the string based on a pattern
    cleaned_list = string_list[1:-1]  # Remove the square brackets
    pattern = r',(?="[^"]+"-tx-\d+-")'
    list_elements = re.split(pattern, cleaned_list)
    return list_elements


# Function to extract trigger word
def extract_trigger_word(trigger_str):
    '''
    To extract trigger word from metamap trigger output (i.e. extract second quoted word)
    '''
    return re.findall(r'\"([^\"]+)\"', trigger_str)[1]  # Extracts the second quoted word


if __name__ == '__main__':
    # example
    from hydra import initialize, compose

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name="config")
    res = convert_cui_to_hpo(cui='C0235031', cfg=cfg)
    print(res)

    res = extract_cuis(
        note="Reason for Request: Hypertonia and poor feeding. Filey is a 5 month old child "
             "who was investigated as an infant for a combination of failure to thrive, pallor "
             "and developmental delay", cfg=cfg)
    print(res)
    print(res.columns)
    print(res['cui'])


# Define a custom function to process the strings
def replace_brackets_pos_info(s):
    if '[' in s:
        # If '[' is present, remove '[', ']' and replace ',' with ';'
        return s.replace('[', '').replace(']', '').replace(',', ';')
    else:
        # Otherwise, don't change the string
        return s
