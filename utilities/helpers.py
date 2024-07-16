import re
import numpy as np

def parse_response(response):
    """
        Parse the textual response to extract claims and associated arguments with their evidence.
        This function is designed to work with responses that are structured in a specific format,
        typically generated by GPT models.

        Args:
        response (str): A multiline string text containing claims and arguments.

        Returns:
        dict: A dictionary containing a 'claim' and a list of 'arguments', each with associated 'evidence'.
        """
    parsed_response = {}
    arguments = []

    # match Claim
    claim_match = re.search(r'\*\*Claim:\*\*(.*?)\n', response, re.S)
    if claim_match:
        parsed_response['claim'] = claim_match.group(1).strip()

    # Matching Arguments and their evidence.
    # Adjust the regular expression to match one or two asterisks, with the colon optionally followed by a space
    argument_matches = re.findall(r'\*{1,2}Argument \d+:\*{0,2} ?(.*?)(?=\*\*|$)(.*?)(?=\n\s*\n|\Z)', response, re.S)
    for argument_match in argument_matches:
        argument = {'title': argument_match[0].strip(), 'evidence': []}
        evidence_matches = re.findall(r'- \*{0,2}Evidence \d+:\*{0,2} (.*?)($|\n)', argument_match[1], re.S)
        for evidence in evidence_matches:
            argument['evidence'].append(evidence[0].strip())
        arguments.append(argument)

    parsed_response['arguments'] = arguments
    return parsed_response

def safe_average(scores):
    if scores:
        # Convert NaN to 0 using np.nan_to_num
        clean_scores = np.nan_to_num(scores)
        return np.mean(clean_scores)
    else:
        return 0.0

def safe_convert_to_numeric(scores):
    # Convert a non-number to NaN and replace it with a 0
    numeric_scores = [np.nan if not isinstance(score, (int, float)) else score for score in scores]
    return np.nan_to_num(numeric_scores)