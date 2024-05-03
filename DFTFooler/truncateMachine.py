import jsonlines

def truncate_text(text, max_length=512):
    """ Truncate text to a maximum length, without cutting words if possible. """
    if len(text) <= max_length:
        return text
    # Try to prevent cutting words in half
    return text[:max_length].rsplit(' ', 1)[0]

def filter_and_truncate_jsonl(input_file, output_file, label_filter='machine', max_length=512, max_samples=100):
    """ Filter JSONL file entries by label, truncate text fields, and limit the output to max_samples. """
    count = 0
    with jsonlines.open(input_file) as reader, jsonlines.open(output_file, mode='w') as writer:
        for obj in reader:
            if obj.get('label') == label_filter:
                obj['text'] = truncate_text(obj['text'], max_length)
                # Add 'split' and 'orig_split' with default values
                obj['split'] = 'test'  # Assuming all data in the file is for testing
                obj['orig_split'] = 'gen'  # Assuming all data is generated
                writer.write(obj)
                count += 1
                if count >= max_samples:
                    break

# Example usage
input_path = '../HuggingfaceData/Data/test_data.jsonl' # Update with your actual file path
output_path = 'df_100_correct_512truncated.jsonl'
filter_and_truncate_jsonl(input_path, output_path)
