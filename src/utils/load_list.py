# To load a word list from a file
def load_word_list(file_path: str) -> list:
    '''
    Load a word list from a file.
    Args:
        file_path (str): Path to the file containing the word list.
    Returns:
        list: List of words loaded from the file.
    '''
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]
