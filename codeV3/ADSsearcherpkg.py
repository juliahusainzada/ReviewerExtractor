import requests
from urllib.parse import urlencode
import numpy as np
import pandas as pd
import TextAnalysis as TA
import itertools
import time

# Define constants
ADS_SEARCH_URL = "https://api.adsabs.harvard.edu/v1/search/query"
ADS_RATE_LIMIT = 0.2
BATCH_SIZE = 25

# Global tracker to prevent redundant author lookups in deep dives
AUTHOR_LOOKUP_CACHE = set()

def chunk_list(data_list, size):
    """
    Return pieces of data_list with n items each
    """
    for i in range(0, len(data_list), size):
        yield data_list[i:i + size]

def do_search(input_name, input_inst, auth_token, query):
    """
    Runs ADS search based on specified query built in ads_search.
    
    Returns a dataframe with the results of the search for a given author or institution.
    """
    time.sleep(ADS_RATE_LIMIT)
    
    # ends an HTTP GET to the ADS search endpoint using the query string q and authenticates with a Bearer token t
    results = requests.get(
        "https://api.adsabs.harvard.edu/v1/search/query?{}".format(query),
        headers={'Authorization': 'Bearer ' + auth_token}
    )

    # Takes results as json
    try:
        json_data = results.json()
    except ValueError:
        time.sleep(ADS_RATE_LIMIT)
        try:
            json_data = results.json()
        except ValueError:
            print("ADS API returned non‑JSON response:", results.status_code, results.text[:200])
            return pd.DataFrame()
    
    # Docs contains each paper that matches the search query
    # We extract relevant fields to create a DataFrame
    data = json_data["response"]["docs"] 

    df_data = {
        'Input Author': [input_name] * len(data),
        'Input Institution': [input_inst] * len(data),
        'First Author': [d['first_author'] for d in data],
        'Bibcode': [d['bibcode'] for d in data],
        'Title': [d.get('title', '') for d in data],
        'Publication Date': [d['pubdate'] for d in data],
        'Keywords': [d.get('keyword', []) for d in data],
        'Affiliations': [d['aff'][0] for d in data],
        'Abstract': [d.get('abstract', '') for d in data],
        'Identifier': [d.get('identifier', []) for d in data],
        'Data Type': ['']*len(data)
    }

    df = pd.DataFrame(df_data)

    # If no input name was provided, use the discovered first author
    # Input name may not be provided in institution searches,
    # so we want to preserve the discovered first author as the input author
    if input_name is None:
        df['Input Author'] = df['First Author']
    
    return df

def format_year(year):
    """
    Standardizes year input into an ADS-compatible [YYYY TO YYYY] string
    """
    # If already a range, return as is 
    if isinstance(year, str) and "TO" in year:
        return year

    # Try to parse as a number and convert to range
    try:
        base_year = int(float(year))
        if base_year - 1 >= 2010:
            print("Warning: Your year input corresponds to a range starting at or after 2010. Early-career classification may be inaccurate.")
        return f"[{base_year - 1} TO {base_year + 4}]"
    
    except ValueError:
        raise ValueError("Invalid year format. Please provide a 4-digit year or a range in the format [YYYY TO YYYY].")

def ads_search(name=None, institution=None, year=None, refereed='property:notrefereed OR property:refereed', \
               token=None, stop_dir=None, second_auth=False,groq_analysis=False,deep_dive=False, early_career=None):
    """
    Builds a query for ADS search based on name, institution, year, second_author.
    
    Builds with a Global AUTHOR_LOOKUP_CACHE to prevent redundant author lookups.

    Returns a dataframe with the results of the search for a given author or institution, 
    including merged results for authors across institutions and n-gram analysis of abstracts.
    """
    global AUTHOR_LOOKUP_CACHE
    query_parts = []

    # ---------------- 1. Building the Query ----------------
    # We only build the query on the name if it's not a deep dive institution search, 
    # otherwise we will search by institution and then deep dive by author name
    if name and not deep_dive:
        if second_auth:
            query_parts.append(f'(first_author:"{name}" OR pos(author:"{name}",2))')
        else:
            query_parts.append(f'first_author:"^{name}"')
    
    if institution:
        query_parts.append(f'pos(institution:"{institution}",1)')
    
    if year:
        years = format_year(year)
        query_parts.append(f'pubdate:{years}')
    
    # Give warning if year range starts at or after 2010 since early-career classification will be inaccurate without earlier publication data
    start_year = int(year.strip('[]').split(' TO ')[0])
    if start_year >= 2010:
        print("Warning: Your year range starts at or after 2010. Early-career classification may be inaccurate.")
    
    if not query_parts:
        print("You did not give me enough to search on, please try again.")
        return pd.DataFrame()
    
    # We call it base query because for deep dives, we will first search by institution 
    # and then build author queries on top of that base query
    base_query = " AND ".join(query_parts)

    # ---------------- 2. Deep Dive Logic ----------------
    # Scout for authors at an institution first
    if institution and deep_dive:
        AUTHOR_LOOKUP_CACHE = set()
        print(f"Step 1: Scouting author names for {institution}...")
        
        # Light search to only get author names
        discovery_params = {
            "q": base_query,
            "fl": "first_author",
            "fq": "database:astronomy," + str(refereed),
            "rows": 3000,
        }

        res = requests.get(
            f"https://api.adsabs.harvard.edu/v1/search/query?{urlencode(discovery_params)}",
            headers={'Authorization': 'Bearer ' + token}
        ).json()

        if "response" in res and res["response"]["docs"]:
            unique_authors = {p.get('first_author') for p in res["response"]["docs"] if p.get('first_author')}
            new_authors = [a for a in unique_authors if a not in AUTHOR_LOOKUP_CACHE]

            print(f"Deep Diving {len(unique_authors)} unique authors...")
            all_author_dfs = []

            for author_batch in chunk_list(new_authors, BATCH_SIZE):
                # Build a query for this batch of authors
                # Create an OR query for all authors in the batch
                author_sub_query = " OR ".join([f'first_author:"{a}"' for a in author_batch])
                batch_query = [f"({author_sub_query})"]

                if year:
                    years = format_year(year)
                    batch_query.append(f'pubdate:{years}')

                batch_query = " AND ".join(batch_query)

                encoded_batch_query = urlencode({
                    "q": batch_query,
                    "fl": "title, first_author, bibcode, abstract, aff, pubdate, keyword, identifier",
                    "fq": "database:astronomy," + str(refereed),
                    "rows": 3000,
                    "sort": "date desc"
                })

                df_batch = do_search(None, institution, token, encoded_batch_query)
                
                if not df_batch.empty:
                    all_author_dfs.append(df_batch)
                
                AUTHOR_LOOKUP_CACHE.update(author_batch)
            
            # AFTER all batches:
            if all_author_dfs:
                full_df = pd.concat(all_author_dfs, ignore_index=True)
                return process_results(full_df, stop_dir, early_career)
            else:
                return pd.DataFrame()
            
    # ---------------- 3. Standard Search Logic ----------------
    # Note: Not deep dive
    encoded_query = urlencode({
        "q": base_query,
        "fl": "title, first_author, bibcode, abstract, aff, pubdate, keyword, identifier",
        "fq": "database:astronomy," + str(refereed),
        "rows": 3000,
        "sort": "date desc"
    })

    results_df = do_search(name, institution, token, encoded_query)

    if not results_df.empty:
       return process_results(results_df, stop_dir, early_career)
    else:
        print("No results found.")
        return pd.DataFrame()

def process_results(df, stop_dir, early_career=None):
    """
    Post-process the search results.
    
    Steps: Merge → data_type → early_career_flag → filter by early_career (optional) → n_grams
    
    Args:
        df: DataFrame with raw search results
        stop_dir: Directory path for stopword loading in n_grams
        early_career: Filter results to early_career=True/False (None = no filter)
    
    Returns:
        Processed DataFrame with all steps applied
    """
    if df.empty:
        return pd.DataFrame()
    
    df = merge(df)
    df = data_type(df)
    df = apply_early_career_flag(df)
    
    if early_career is not None:
        df = df[df['Early Career'] == early_career]
    
    df = compute_n_grams(df, stop_dir)
    
    return df

def data_type(df):
    """
    Determines whether at least half of the author's publications are in the specified list of journals. 
    
    Returns the dataframe with the 'Data Type' column added with the label 'Clean' or 'Dirty'. 

    Labels authors as 'Clean' if >50% of papers are in core astronomy journals.
    """
    journals = ['ApJ','GCN','MNRAS', 'AJ', 'Nature', 'Science', 'PASP', 'AAS', 'arXiv', 'SPIE', 'A&A', 'zndo','yCat','APh', 'PhRvL']
    df['Data Type'] = ''

    # For each author...
    for index, row in df.iterrows():
        bibcodes_str = row['Bibcode']
        # Split the Bibcode string into individual bibcodes,
        bibcodes = bibcodes_str.split(', ')

        # Check how many are in the specified journals
        total_papers = len(bibcodes)
        clean_count = sum(any(journal in bibcode for journal in journals) for bibcode in bibcodes)
        
        # Label as 'Clean' or 'Dirty'
        if clean_count >= total_papers / 2:
            data_type_label = 'Clean'
        else:
            data_type_label = 'Dirty'
        df.at[index, 'Data Type'] = data_type_label
    
    return df
        
def merge(df):
    """
    Merges all rows under the same author name and concatenates their results.
    
    Returns the resulting merged dataframe.
    """
    df['Publication Date'] = df['Publication Date'].astype(str)
    df['Abstract'] = df['Abstract'].astype(str)

    df['Keywords'] = df['Keywords'].apply(lambda keywords: keywords if keywords else [])
    df['Title'] = df['Title'].apply(lambda titles: titles if titles else []) 
    df['Identifier'] = df['Identifier'].apply(lambda ids: ids if ids else []) 
    
    df.fillna('None', inplace=True)

    merged = df.groupby('Input Author').aggregate({'Input Institution': lambda x: ", ".join(sorted(set(x))),
                                                 'First Author': ', '.join,
                                                 'Bibcode': ', '.join,
                                                 'Title': lambda x: list(itertools.chain.from_iterable(x)), # become one big list
                                                 'Publication Date': ', '.join,
                                                 'Keywords': lambda x: list(itertools.chain.from_iterable(x)), # <- Fix for Keywords
                                                 'Affiliations': ', '.join,
                                                 'Abstract': ', '.join,
                                                 'Data Type': ', '.join,
                                                 'Identifier': lambda x: list(itertools.chain.from_iterable(x))  
                                                 }).reset_index()
    return merged

def compute_n_grams(df, stop_words_path):
    """
    Calculates the top words, bigrams, and trigrams for through an author's abstracts.
    
    Returns the dataframe including the top 10 words, bigrams, and trigrams.
    """
    top_words, top_bigrams, top_trigrams = [], [], []

    stop_words = TA.stopword_loader(stop_words_path)

    for abstract in df['Abstract']:
        tokens = TA.preprocess_text(abstract, stop_words)
        top_words.append(TA.compute_top_ngrams(tokens, n=1))
        top_bigrams.append(TA.compute_top_ngrams(tokens, n=2))
        top_trigrams.append(TA.compute_top_ngrams(tokens, n=3))

    df = df.copy()
    df['Top 10 Words'] = top_words
    df['Top 10 Bigrams'] = top_bigrams
    df['Top 10 Trigrams'] = top_trigrams
    return df

def apply_early_career_flag(df, cutoff_year=2010):
    """
    Flags whether an author is early career using the merged Publication Date column

    Early career = no publication prior to cutoff_year
    """

    early_flags = []

    for dates in df['Publication Date']:
        # Split comma-separated pubdates
        date_list = [d.strip() for d in dates.split(",")]

        # Extract year from 'YYYY-MM' or 'YYYY'
        years = []
        for d in date_list:
            if len(d) >= 4:
                year = int(d[:4])
                years.append(year)
            
        # Find earliest year
        earliest_year = min(years)

        # Early career condition
        early_flags.append(earliest_year >= cutoff_year)

    df = df.copy()
    df['Early Career'] = early_flags
    return df

def get_user_input(dataframe):
    """
    Gets user input for searching a dataframe.
    
    Returns a dictionary with search parameters for either a name or institution search.
    """

    # Helper: Handles Yes/No/None 
    def ask_yes_no(prompt, default="n"):
        if default is None:
            hint = "(y/n) [Press Enter for None]"
        else:
            hint = f"(y/n) [Default: {default}]" 
        
        choice = input(f"{prompt} {hint}: ").strip().lower()

        if not choice:
            return default
    
        if choice == 'y': return True
        if choice == 'n': return False 

        return default
    
    # Helper: Matches user string to actual dataframe columns (case-insensitive)
    def find_column(prompt, default_col):
        column_map = {c.lower(): c for c in dataframe.columns} 

        while True:
            user_input = input(f"{prompt} [Default: {default_col}]: ").strip()

            target = user_input.lower() if user_input else default_col.lower()

            match = column_map.get(target)

            if match:
                return match

            # If we get here, the input (or the default) wasn't found
            print(f"\nError: '{target}' not found in your file.")
            print(f"Available columns: {', '.join(dataframe.columns)}")
            print("Please try again or check your spelling.\n")
    
    # Define available search types for user selection
    available_search_types = {
        "name": "Name Search - search by author name",
        "institution": "Institution Search - search by institution"
    }
    
    # 1. Select search type
    print("\nWhat type of search do you want to conduct?")
    for key, description in available_search_types.items():
        print(f"-Enter '{key}' for {description}")
    
    search_type = ""
    while search_type not in available_search_types:
        search_type = input("\nEnter search type: ").lower().strip()
    
    # 2. Create search_params dict to hold parameters for the ADS search query
    search_params = {'search_type': search_type}
    print(f"\nAvailable columns: {', '.join(dataframe.columns)}")
    
    if search_type == 'name':
        search_params['name_column'] = find_column("Enter the name of the column that contains the data for 'name' search: ", "Name")

    else:
        search_params['institution_column'] = find_column("Enter the name of the column that contains the data for 'institution' search: ", "Name")
        search_params['deep_dive'] = ask_yes_no("Do you want to run a deep dive search (re-run for each author) for institution search?", default="n")

    search_params['second_author'] = ask_yes_no("Do you want to include search by second author? (y/n) [n]: ", default="n") 
    
    # 3. Year and filter options
    print("\nNOTE:")
    print("Early-career classification depends on the publication history returned by ADS.")
    print("If the selected year range does not include years prior to 2010, the system")
    print("cannot determine whether an author had earlier publications.")
    print("This may cause senior researchers to be incorrectly flagged as early-career.\n")

    year_range = input("Enter the year range for your search (format: [YYYY TO YYYY] or a 4-digit year, default: [2003 TO 2030]): ").strip() or "[2003 TO 2030]"
    search_params['year_range'] = year_range
    
    is_refereed = ask_yes_no("Do you want refereed papers only? (y/n) [y]:", default="y")
    search_params['refereed'] = "property:refereed" if is_refereed else "property:notrefereed OR property:refereed"
    
    search_params['early_career'] = ask_yes_no("Filter for early-career researchers only?", default=None)    
    
    return search_params

def run_file_search(filename,  token, stop_dir, year=None, second_auth=False,
                        refereed='property:notrefereed OR property:refereed'):
    """
    Runs ADS search based on user's search type (name or institution).
    
    Ensures authors found across multiple institutions are merged into a single row
    with an aggregated institution list.
    """
    # --------- 1. Load data and get user search parameters ---------
    try:
        raw_data = pd.read_csv(filename, quotechar='"')
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return pd.DataFrame()

    search_params = get_user_input(raw_data)
    
    all_results = []
    search_type = search_params['search_type']

    # Identify which column we are iterating over
    target_col = search_params.get('name_column') if search_type == 'name' else search_params.get('institution_column')
    
    print(f"\nStarting {search_type} search for {len(raw_data)} rows...")

    # --------- 2. Process each row in the CSV --------- 
    for index, row in raw_data.iterrows():
        search_val = str(row.get(target_col, "")).strip().strip('"')
        if not search_val or search_val.lower() == "nan":
            continue
    
        print(f"[{index + 1}/{len(raw_data)}] Searching for: {search_val}")

        # Prepare arguments for ads_search based on type
        search_args = {
            'year': search_params['year_range'],
            'token': token,
            'stop_dir': stop_dir,
            'second_auth': search_params['second_author'],
            'refereed': search_params['refereed'],
            'early_career': search_params['early_career'],  # Pass the True/False/None value
            'deep_dive': search_params.get('deep_dive', False)
        }

        if search_type == 'name':
            search_args['name'] = search_val
            search_args['institution'] = None
        else:
            search_args['name'] = None
            search_args['institution'] = search_val

        # Execute search
        result_df = ads_search(**search_args)
    
        if not result_df.empty:
            all_results.append(result_df)
        
        time.sleep(ADS_RATE_LIMIT)
    
    # --------- 3. Combine results and post-process ---------
    if not all_results:
        print("No results found for any search terms.")
        return pd.DataFrame()

    final_df = pd.concat(all_results, ignore_index=True)
    final_df = process_results(final_df, stop_dir, search_params['early_career'])
        
    print(f"Search complete. {len(final_df)} unique author records found.")
    return final_df