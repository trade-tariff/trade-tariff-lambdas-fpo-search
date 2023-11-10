import csv

def read_description_csv(file_path):

    with open(file_path, mode='r', encoding='Windows-1252') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)  # skip the first line
        data_list = list(csv_reader)

    print(f'Read {len(data_list)} rows of data, excluding the header. Type is {type(data_list)}')

    return data_list

def read_multiple_csvs(file_paths):
    data_list = []
    for file_path in file_paths:
        print(f"Reading {file_path}:")
        with open(file_path, mode='r', encoding='Windows-1252') as csv_file:
            csv_reader = csv.reader(csv_file)
            header = next(csv_reader)  # skip the first line
            data_list += list(csv_reader)

    print(f'Read {len(data_list)} rows of data, excluding the header.')

    return data_list

# so we need to group these entries. Each line looks like this:
#  85334010 00,ELECTRONIC COMPONENTS
# we want a hash keyed on 85334010 with the value to be an array of strings
def group_description_data(data_list):

    invalid_lines = 0
    grouped_data = {}
    for level in range(2, 10, 2):
        grouped_data[level] = {}

    for line in data_list:
        # print(f"line = {line}")
        # split the line - 8 digits, space, 2 digits, comma, free text
        # Minimal validation
        if len(line) != 2 or len(line[0]) != 11 or len(line[1]) < 1:
            # print(f"Invalid line: {line}")
            invalid_lines += 1
            continue
 
        keybits = line[0].split(' ')
        if len(keybits) != 2 or len(keybits[0]) != 8:
            # print(f"Invalid key: {keybits}")
            invalid_lines += 1
            continue

        eightkey = keybits[0]
        description = line[1].strip()

        # print(f"key = {eightkey}, description = {description}")
        for level in range(2, 10, 2):
            key = eightkey[:level]
            if key in grouped_data[level]:
                grouped_data[level][key].append(description)
            else:
                grouped_data[level][key] = [description]
    
    print(f"Invalid lines: {invalid_lines}")
    return grouped_data

# grouped_data = group_description_data(read_description_csv('source_data/DEC22COMCODEDESCRIPTION.csv'))
# print(len(grouped_data), len(grouped_data[2]), len(grouped_data[4]), 
#       len(grouped_data[6]),  len(grouped_data[8]))
# print(grouped_data)

def get_commodities_terms():
    code_data = read_description_csv("source_data/uk_commodities_2023-06-22.csv")
    mapped_codes = {}
    # build map of lines by code first, in case some are out of order
    for line in code_data:
        mapped_codes[line[9]] = line

    commodities_descriptions = {}
    for level in range(2, 10, 2):
        commodities_descriptions[level] = {}

    # go back through the codes, building a text string from all the ancestors for each
    processed_codes = 0
    ignored_codes = 0
    for line in code_data:
        # only index the "end line" entries, for now? // TODO <- revisit this decision
        if line[6] != "1":
            ignored_codes += 1
            continue

        processed_codes += 1

        code = line[9]
        ancestor_codes = line[10].split(',')
        description = line[7]
        for ancestor_code in ancestor_codes:
            ancestor = mapped_codes[ancestor_code]
            description = ancestor[7] + " " + description

        for level in range(2, 10, 2):
            key = code[:level]
            if key in commodities_descriptions[level]:
                commodities_descriptions[level][key].add(description)
            else:
                commodities_descriptions[level][key] = {description}
    
    print(f"Processed {processed_codes} codes, ignored {ignored_codes} codes")

    return commodities_descriptions
    
def read_fpo_classified_data(filename, description_col, code_col):
    input_data = read_description_csv(filename)
    classified_data = []
    for line in input_data:
        description = line[description_col].strip()
        code = line[code_col].strip()
        # so we're ignoring the ones with either a blank description or blank code for now
        # but maybe in the future we can try classifying the ones with a blank code
        # and sending them back to the classifiers to see how well/badly we did.
        if len(description) > 0 and len(code) > 0:
            classified_data.append((description, code))
    
    print(f"Extracted {len(classified_data)} usable rows of classified FPO data")
    return classified_data