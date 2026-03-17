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
BATCH_SIZE = 40

# Global tracker to prevent redundant author lookups in deep dives
AUTHOR_LOOKUP_CACHE = set()

def make_ads_request(url, headers, max_retries=5):
    """
    Wrapper for requests.get to cleanly handle ADS API rate limits (HTTP 429).
    """
    retries = 0
    while retries < max_retries:
        time.sleep(ADS_RATE_LIMIT)
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            try:
                return response.json()
            except ValueError:
                print("ADS API returned non-JSON response:", response.status_code, response.text[:200])
                return None
        elif response.status_code == 429:
            # Rate limited
            reset_time_str = response.headers.get('X-RateLimit-Reset')
            if reset_time_str:
                try:
                    reset_timestamp = int(reset_time_str)
                    current_time = int(time.time())
                    sleep_duration = max(reset_timestamp - current_time, 1) + 1 # +1 buffer
                    print(f"Rate limited (429). Sleeping for {sleep_duration} seconds...")
                    time.sleep(sleep_duration)
                except ValueError:
                    print("Could not parse X-RateLimit-Reset header. Sleeping for 60 seconds...")
                    time.sleep(60)
            else:
                print("Rate limited but no X-RateLimit-Reset header found. Sleeping for 60 seconds...")
                time.sleep(60)
            retries += 1
        else:
            print(f"ADS API error: {response.status_code} - {response.text[:200]}")
            return None
            
    print("Max retries exceeded for ADS API.")
    return None

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
    url = f"https://api.adsabs.harvard.edu/v1/search/query?{query}"
    headers = {'Authorization': 'Bearer ' + auth_token}
    
    json_data = make_ads_request(url, headers)
    
    if json_data is None or "response" not in json_data or "docs" not in json_data["response"]:
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

def ads_search(name=None, institution=None, year="[2003 TO 2030]", refereed='property:notrefereed OR property:refereed', \
               token=None, stop_dir=None, second_auth=False, deep_dive=False, early_career=False, \
                filename=None, search_type=None, institution_column=None, name_column=None, process=True, clear_cache=True):
    """
    Builds a query for ADS search based on name, institution, year, second_author.
    If filename is provided, it iterates through the file and performs a batch search.

    Builds with a Global AUTHOR_LOOKUP_CACHE to prevent redundant author lookups.

    Returns a dataframe with the results of the search for a given author or institution, 
    including merged results for authors across institutions and n-gram analysis of abstracts.
    """
    global AUTHOR_LOOKUP_CACHE
    if clear_cache:
        AUTHOR_LOOKUP_CACHE = set()

    # ---------------- 1. Building the Query ----------------
    if filename:
        try:
            raw_data = pd.read_csv(filename, quotechar='"')
            # Determine search type and target column
            if search_type.lower() == 'institution':
                target_col = institution_column or "Institution"
            else:
                target_col = name_column or "Name"
            
            all_results = []
            failed_searches = []
            fallback_institutions = []
            print(f"Processing file: {filename} ({len(raw_data)} rows)...")

            if search_type.lower() == 'institution' and deep_dive:
                all_unique_authors = set()
                author_to_institutions = {}
                
                # --- Phase 1: Global Scouting ---
                print(f"\n--- Phase 1: Institutional Scouting ---")
                for index, row in raw_data.iterrows():
                    search_val = str(row.get(target_col, "")).strip().strip('"')
                    if not search_val or search_val.lower() == "nan":
                        continue
                        
                    print(f"[{index + 1}/{len(raw_data)}] Scouting institution: {search_val}")
                    
                    query_parts = [f'pos(institution:"{search_val}",1)']
                    if year:
                        query_parts.append(f'pubdate:{format_year(year)}')
                    base_query = " AND ".join(query_parts)
                    
                    discovery_params = {
                        "q": base_query,
                        "fl": "first_author",
                        "fq": "database:astronomy," + str(refereed),
                        "rows": 3000,
                    }
                    url = f"https://api.adsabs.harvard.edu/v1/search/query?{urlencode(discovery_params)}"
                    headers = {'Authorization': 'Bearer ' + token}
                    res = make_ads_request(url, headers)
                    
                    if not res or "response" not in res or not res["response"]["docs"]:
                        print(f"  No results. Retrying with affiliation fallback...")
                        fallback_institutions.append(search_val)
                        discovery_params["q"] = base_query.replace(f'pos(institution:"{search_val}",1)', f'pos(aff:"{search_val}",1)')
                        url = f"https://api.adsabs.harvard.edu/v1/search/query?{urlencode(discovery_params)}"
                        res = make_ads_request(url, headers)
                        
                    if res and "response" in res and res["response"]["docs"]:
                        unique_authors = {p.get('first_author') for p in res["response"]["docs"] if p.get('first_author')}
                        for author in unique_authors:
                            all_unique_authors.add(author)
                            if author not in author_to_institutions:
                                author_to_institutions[author] = []
                            author_to_institutions[author].append(search_val)
                    else:
                        print(f"  Fallback failed. Adding {search_val} to failed searches.")
                        failed_searches.append(search_val)

                # --- Phase 2: Global Batch Fetching ---
                new_authors = [a for a in all_unique_authors if a not in AUTHOR_LOOKUP_CACHE]
                
                print(f"\n--- Phase 2: Batch Deep Dive Fetching ({len(new_authors)} unique authors globally) ---")
                all_author_dfs = []
                author_batches = list(chunk_list(new_authors, BATCH_SIZE))
                
                for batch_idx, author_batch in enumerate(author_batches):
                    print(f"Processing batch {batch_idx + 1}/{len(author_batches)} ({len(author_batch)} authors)...")
                    author_sub_query = " OR ".join([f'first_author:"{a}"' for a in author_batch])
                    batch_query = [f"({author_sub_query})"]
                    
                    if year:
                        batch_query.append(f'pubdate:{format_year(year)}')
                    batch_query = " AND ".join(batch_query)
                    
                    encoded_batch_query = urlencode({
                        "q": batch_query,
                        "fl": "title, first_author, bibcode, abstract, aff, pubdate, keyword, identifier",
                        "fq": "database:astronomy," + str(refereed),
                        "rows": 3000,
                        "sort": "date desc"
                    })
                    
                    df_batch = do_search(None, None, token, encoded_batch_query)
                    if not df_batch.empty:
                        all_author_dfs.append(df_batch)
                        
                    AUTHOR_LOOKUP_CACHE.update(author_batch)

                if fallback_institutions:
                    print("\n--- FALLBACK INSTITUTIONS ---")
                    print(f"The following {len(fallback_institutions)} items had no institution results and fell back to affiliation:")
                    for item in fallback_institutions:
                        print(f"{item}")
                    print("-----------------------------\n")

                if failed_searches:
                    print("\n--- SEARCH FAILURES ---")
                    print(f"The following {len(failed_searches)} items returned NO results:")
                    for item in failed_searches:
                        print(f"{item}")
                    print("-----------------------\n")
                    
                if all_author_dfs:
                    final_df = pd.concat(all_author_dfs, ignore_index=True)
                    def assign_insts(author):
                        insts = author_to_institutions.get(author, [])
                        return ", ".join(insts) if insts else None
                    final_df['Input Institution'] = final_df['First Author'].apply(assign_insts)
                    
                    if process:
                        return process_results(final_df, stop_dir, early_career)
                    else:
                        return final_df
                else:
                    return pd.DataFrame()

            # --------- Standard Batch Iteration --------- 
            for index, row in raw_data.iterrows():
                search_val = str(row.get(target_col, "")).strip().strip('"')
                if not search_val or search_val.lower() == "nan":
                    continue

                print(f"[{index + 1}/{len(raw_data)}] Searching for: {search_val}")

                row_df = ads_search(
                    name=search_val if search_type.lower() == 'name' else None,
                    institution=search_val if search_type.lower() == 'institution' else None,
                    year=year,
                    refereed=refereed,
                    token=token,
                    stop_dir=stop_dir,
                    second_auth=second_auth,
                    deep_dive=deep_dive,
                    early_career=early_career,
                    process=False,
                    clear_cache=False # Maintain internal cache during standard iteration
                )

                if not row_df.empty:
                    all_results.append(row_df)
                else:
                    failed_searches.append(search_val)
                
                time.sleep(ADS_RATE_LIMIT)
            
            if fallback_institutions:
                print("\n--- FALLBACK INSTITUTIONS ---")
                print(f"The following {len(fallback_institutions)} items had no institution results and fell back to affiliation:")
                for item in fallback_institutions:
                    print(f"{item}")
                print("-----------------------------\n")
            
            if failed_searches:
                print("\n--- SEARCH FAILURES ---")
                print(f"The following {len(failed_searches)} items returned NO results:")
                for item in failed_searches:
                    print(f"{item}")
                print("-----------------------\n")
            
            if not all_results:
                return pd.DataFrame()
            
            final_df = pd.concat(all_results, ignore_index=True)
            if process:
                return process_results(final_df, stop_dir, early_career)
            else:
                return final_df

        except FileNotFoundError:
            print(f"Error: The file '{filename}' was not found.")
            return pd.DataFrame()
            
    # ---------------- 1. Building the Query ----------------
    # We only build the query on the name if it's not a deep dive institution search, 
    # otherwise we will search by institution and then deep dive by author name
    query_parts = []

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
        # AUTHOR_LOOKUP_CACHE = set()
        print(f"Step 1: Scouting author names for {institution}...")
        
        # Light search to only get author names
        discovery_params = {
            "q": base_query,
            "fl": "first_author",
            "fq": "database:astronomy," + str(refereed),
            "rows": 3000,
        }

        url = f"https://api.adsabs.harvard.edu/v1/search/query?{urlencode(discovery_params)}"
        headers = {'Authorization': 'Bearer ' + token}
        res = make_ads_request(url, headers)

        if not res or ("response" not in res or not res["response"]["docs"]):
            print(f"No results for institution '{institution}'. Retrying with affiliation fallback...")
            discovery_params["q"] = base_query.replace(f'pos(institution:"{institution}",1)', f'pos(aff:"{institution}",1)')
            url = f"https://api.adsabs.harvard.edu/v1/search/query?{urlencode(discovery_params)}"
            res = make_ads_request(url, headers)

        if res and "response" in res and res["response"]["docs"]:
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
                if process:
                    return process_results(full_df, stop_dir, early_career)
                else:
                    return full_df
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

    if results_df.empty and institution and not deep_dive:
        print(f"No results for institution '{institution}'. Retrying with affiliation fallback...")
        fallback_query = base_query.replace(f'pos(institution:"{institution}",1)', f'pos(aff:"{institution}",1)')
        encoded_fallback = urlencode({
            "q": fallback_query,
            "fl": "title, first_author, bibcode, abstract, aff, pubdate, keyword, identifier",
            "fq": "database:astronomy," + str(refereed),
            "rows": 3000,
            "sort": "date desc"
        })
        results_df = do_search(name, institution, token, encoded_fallback)

    if not results_df.empty:
       if process:
           return process_results(results_df, stop_dir, early_career)
       else:
           return results_df
    else:
        print("No results found.")
        return pd.DataFrame()

def process_results(df, stop_dir, early_career=False):
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
    
    if early_career is True:
        df = df[df['Early Career'] == True]
    
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
        hint = f"(y/n) [Default: {default}]" 
        
        choice = input(f"{prompt} {hint}: ").strip().lower()

        if not choice:
            return default
    
        if choice == 'y': 
            return True
        else:
            return False
    
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
    print("Available search types:")
    for key, description in available_search_types.items():
        print(f"-Enter '{key}' for {description}")

    while True:
        try:
            search_type = input("\nEnter search type: ('name' or 'institution'):\n").lower()
            if search_type in available_search_types:
                break
            print("Invalid search type. Please enter 'name' or 'institution'.")
        except NameError:
            print("Error getting input. Please try again.")
    
    print(f"✓ '{search_type}' search.")
    
    # 2. Create search_params dict to hold parameters for the ADS search query
    search_params = {'search_type': search_type}
    print(f"\nThese are the available columns from your dataset: {', '.join(dataframe.columns)}")
    
    if search_type == 'name':
        search_params['name_column'] = find_column("Enter the name of the column that contains the data for 'name' search: ", "Name")
        search_params['institution_column'] = None
        print(f"✓ '{search_params['name_column']}' column.")

    else:
        search_params['institution_column'] = find_column("Enter the name of the column that contains the data for 'institution' search: ", "Name")
        print(f"✓ '{search_params['institution_column']}' column.")
        search_params['name_column'] = None
        search_params['deep_dive'] = ask_yes_no("Do you want to run a deep dive search (re-run for each author) for institution search?", default="n")
        print(f"✓ '{search_params['deep_dive']}' for deep dive.")

    search_params['second_author'] = ask_yes_no("Do you want to include search by second author? (y/n) [n]: ", default="n") 
    print(f"✓ '{search_params['second_author']}' for second author search.")
    
    # 3. Year and filter options
    print("\nNOTE:")
    print("Early-career classification depends on the publication history returned by ADS.")
    print("If the selected year range does not include years prior to 2010, the system")
    print("cannot determine whether an author had earlier publications.")
    print("This may cause senior researchers to be incorrectly flagged as early-career.\n")

    year_range = input("Enter the year range for your search (format: [YYYY TO YYYY] or a 4-digit year, default: [2003 TO 2030]): ").strip() or "[2003 TO 2030]"
    search_params['year_range'] = year_range
    print(f"✓ '{search_params['year_range']}' for the year range.")
    
    is_refereed = ask_yes_no("Do you want refereed papers only? (y/n) [y]:", default="y")
    search_params['refereed'] = "property:refereed" if is_refereed else "property:notrefereed OR property:refereed"
    print(f"✓ '{search_params['refereed']}' for refereed papers.")

    search_params['early_career_filter'] = ask_yes_no("Filter results for early-career researchers ONLY?")
    print(f"✓ '{"Yes" if search_params['early_career_filter'] == True else "No"}' for early-career researchers.")   

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
    failed_searches = []
    fallback_institutions = []

    # Map the boolean choice
    # If filter is False (user said 'n'), we pass None
    ec_filter_val = search_params['early_career_filter']

    search_type = search_params['search_type']

    # Identify which column we are iterating over
    # Identify which column we are iterating over
    if search_type == 'name':
        target_col = search_params.get('name_column')
    else:
        target_col = search_params.get('institution_column')
    
    print(f"\nStarting {search_type} search for {len(raw_data)} rows...")

    if search_type == 'institution' and search_params.get('deep_dive', False):
        all_unique_authors = set()
        author_to_institutions = {}
        
        # --- Phase 1: Global Scouting ---
        print(f"\n--- Phase 1: Institutional Scouting ---")
        for index, row in raw_data.iterrows():
            search_val = str(row.get(target_col, "")).strip().strip('"')
            if not search_val or search_val.lower() == "nan":
                continue
                
            print(f"[{index + 1}/{len(raw_data)}] Scouting institution: {search_val}")
            
            query_parts = [f'pos(institution:"{search_val}",1)']
            if search_params['year_range']:
                query_parts.append(f'pubdate:{format_year(search_params["year_range"])}')
            base_query = " AND ".join(query_parts)
            
            discovery_params = {
                "q": base_query,
                "fl": "first_author",
                "fq": "database:astronomy," + str(search_params['refereed']),
                "rows": 3000,
            }
            url = f"https://api.adsabs.harvard.edu/v1/search/query?{urlencode(discovery_params)}"
            headers = {'Authorization': 'Bearer ' + token}
            res = make_ads_request(url, headers)
            
            if not res or "response" not in res or not res["response"]["docs"]:
                print(f"  No results. Retrying with affiliation fallback...")
                fallback_institutions.append(search_val)
                discovery_params["q"] = base_query.replace(f'pos(institution:"{search_val}",1)', f'pos(aff:"{search_val}",1)')
                url = f"https://api.adsabs.harvard.edu/v1/search/query?{urlencode(discovery_params)}"
                res = make_ads_request(url, headers)
                
            if res and "response" in res and res["response"]["docs"]:
                unique_authors = {p.get('first_author') for p in res["response"]["docs"] if p.get('first_author')}
                for author in unique_authors:
                    all_unique_authors.add(author)
                    if author not in author_to_institutions:
                        author_to_institutions[author] = []
                    author_to_institutions[author].append(search_val)
            else:
                print(f"  Fallback failed. Adding {search_val} to failed searches.")
                failed_searches.append(search_val)
                
        # --- Phase 2: Global Batch Fetching ---
        new_authors = [a for a in all_unique_authors if a not in AUTHOR_LOOKUP_CACHE]
        
        print(f"\n--- Phase 2: Batch Deep Dive Fetching ({len(new_authors)} unique authors globally) ---")
        all_author_dfs = []
        author_batches = list(chunk_list(new_authors, BATCH_SIZE))
        
        for batch_idx, author_batch in enumerate(author_batches):
            print(f"Processing batch {batch_idx + 1}/{len(author_batches)} ({len(author_batch)} authors)...")
            author_sub_query = " OR ".join([f'first_author:"{a}"' for a in author_batch])
            batch_query = [f"({author_sub_query})"]
            
            if search_params['year_range']:
                batch_query.append(f'pubdate:{format_year(search_params["year_range"])}')
            batch_query = " AND ".join(batch_query)
            
            encoded_batch_query = urlencode({
                "q": batch_query,
                "fl": "title, first_author, bibcode, abstract, aff, pubdate, keyword, identifier",
                "fq": "database:astronomy," + str(search_params['refereed']),
                "rows": 3000,
                "sort": "date desc"
            })
            
            df_batch = do_search(None, None, token, encoded_batch_query)
            if not df_batch.empty:
                all_author_dfs.append(df_batch)
                
            AUTHOR_LOOKUP_CACHE.update(author_batch)
            
        if fallback_institutions:
            print("\n" + "!"*30)
            print(f"NOTICE: {len(fallback_institutions)} searches returned no institution results and fell back to affiliation.")
            print(f"Fallback items: {', '.join(fallback_institutions)}")
            print("!"*30 + "\n")
            
            # Optional: Save fallbacks to a CSV for manual inspection
            pd.DataFrame(fallback_institutions, columns=['Fallback_Institution']).to_csv("fallback_institutions.csv", index=False)
            
        if failed_searches:
            print("\n" + "!"*30)
            print(f"NOTICE: {len(failed_searches)} searches returned zero results.")
            print("This usually means the institution name is formatted differently in ADS or the year range is too restrictive.")
            print(f"Failed items: {', '.join(failed_searches)}")
            print("!"*30 + "\n")
            
        if all_author_dfs:
            final_df = pd.concat(all_author_dfs, ignore_index=True)
            def assign_insts(author):
                insts = author_to_institutions.get(author, [])
                return ", ".join(insts) if insts else None
            final_df['Input Institution'] = final_df['First Author'].apply(assign_insts)
            
            final_df = process_results(final_df, stop_dir, ec_filter_val)
            print(f"Search complete. {len(final_df)} unique author records found.")
            return final_df
        else:
            print("No results found.")
            return pd.DataFrame()

    # --------- 2. Process each row in the CSV (Standard Pattern) --------- 
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
            'early_career': ec_filter_val,
            'deep_dive': search_params.get('deep_dive', False)
        }

        if search_type == 'name':
            search_args['name'] = search_val
            search_args['institution'] = None
        else:
            search_args['name'] = None
            search_args['institution'] = search_val

        # Execute search without running process_results multiple times
        search_args['process'] = False
        search_args['clear_cache'] = False # Relying on cache management inside run_file_search
        result_df = ads_search(**search_args)
    
        if not result_df.empty:
            all_results.append(result_df)
        else:
            failed_searches.append(search_val)

        time.sleep(ADS_RATE_LIMIT)
    
    if fallback_institutions:
        print("\n" + "!"*30)
        print(f"NOTICE: {len(fallback_institutions)} searches returned no institution results and fell back to affiliation.")
        print(f"Fallback items: {', '.join(fallback_institutions)}")
        print("!"*30 + "\n")

        # Optional: Save fallbacks to a CSV for manual inspection
        pd.DataFrame(fallback_institutions, columns=['Fallback_Institution']).to_csv("fallback_institutions.csv", index=False)
        
    if failed_searches:
        print("\n" + "!"*30)
        print(f"NOTICE: {len(failed_searches)} searches returned zero results.")
        print("This usually means the institution name is formatted differently in ADS or the year range is too restrictive.")
        print(f"Failed items: {', '.join(failed_searches)}")
        print("!"*30 + "\n")

        # Optional: Save failures to a CSV for manual inspection
        pd.DataFrame(failed_searches, columns=['Failed_Search_Term']).to_csv("failed_searches.csv", index=False)

    
    # --------- 3. Combine results and post-process ---------
    if not all_results:
        print("No results found for any search terms.")
        return pd.DataFrame()

    final_df = pd.concat(all_results, ignore_index=True)
    final_df = process_results(final_df, stop_dir, ec_filter_val)
        
    print(f"Search complete. {len(final_df)} unique author records found.")
    return final_df