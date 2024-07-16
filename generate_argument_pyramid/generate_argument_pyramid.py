import openai
from evaluation.evaluation_pyramid import calculate
from utilities.extract_evidence_from_web import extract_evidence_from_web
from config.config import bing_api_key, threshold

def generate_argument_pyramid(question, contents, openai_api_key):
    """
    Generates an argument pyramid using the OpenAI GPT API. This function formats the input, sends it to the API,
    and processes the response to create a structured argument pyramid.

    Args:
    question (str): The question to which the argument pyramid should respond.
    contents (list of str): The list of evidence pieces to support the argument.
    openai_api_key (str): The API key for accessing OpenAI's GPT model.

    Returns:
    str: A structured argument pyramid as a single formatted string.
    """
    openai.api_key = openai_api_key

    # Converts a list of contents into a single string, each separated by two newlines
    formatted_contents = "\n\n".join(contents)
    question = question

    # Defining Prompt Messages
    messages = [
        {"role": "system", "content": "You are a skilled debater and expert in constructing logical arguments."},
        {"role": "user", "content": f"""
        Create an argument pyramid with the following structure based on the evidences provided: "{formatted_contents}".

        1. **Claim:** Provide a claim or thesis statement that directly answers or addresses the question: "{question}", ensuring it is in line with how the question is posed (e.g., using 'should' if the question does).

        2. **Arguments:** Provide at least five complete sentences as arguments supporting the claim, and include evidence for each argument. Ensure each argument is clearly stated and distinct.

        3. **Evidence:** For each argument, provide at least four pieces of evidence to support it. The evidence should be credible, varied (e.g., statistics, expert opinions, real-world examples), and directly relevant to the argument.

        Ensure that the structure follows this format:

        - **Claim:** [Your claim here]
          - **Argument 1: [Argument]**
            - Evidence 1: [Detail of the evidence]
            - Evidence 2: [Detail of the evidence]
            - Evidence 3: [Detail of the evidence]
            - Evidence 4: [Detail of the evidence]
          - **Argument 2: [Another Argument]**
            - Evidence 1: [Detail of the evidence]
            - Evidence 2: [Detail of the evidence]
            - Evidence ...
          - **Argument 3**
            - Evidence 1
            - Evidence 2
            - Evidence ...
          - **Argument ...**
            - Evidence 1
            - Evidence 2
            - Evidence ...

        Please use clear and logical reasoning throughout.
        """}
    ]

    # Send Request
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        #model="gpt-4",
        messages=messages,
        max_tokens=1500,
        n=1,
        temperature=0.7,
    )

    # Extracting response content
    result = response.choices[0].message['content'].strip()

    return result


def improve_argument_pyramid(pyramid, contents, prompt, openai_api_key):
    """
    Enhances an existing argument pyramid by reformatting it with additional content and insights using OpenAI's GPT.
    The function takes an existing pyramid, formats a prompt to enhance it, and sends the request to OpenAI's GPT model.

    Args:
    pyramid (str): The existing argument pyramid text that needs improvement.
    contents (list of str): A list of new evidence or content that should be incorporated into the pyramid.
    prompt (str): The template string that outlines how the pyramid should be improved.
    openai_api_key (str): The API key for accessing OpenAI's GPT model.

    Returns:
    str: The improved argument pyramid as a formatted string.
    """
    openai.api_key = openai_api_key

    formatted_contents = "\n\n".join(contents)

    prompt = prompt.format(
        pyramid=pyramid,
        contents=formatted_contents
    )

    messages = [
        {"role": "system", "content": "You are a skilled debater and expert in constructing logical arguments."},
        {"role": "user", "content": prompt}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        #model="gpt-4",
        messages=messages,
        max_tokens=1500,
        n=1,
        temperature=0.7,
    )

    # Extracting response content
    result = response.choices[0].message['content'].strip()

    return result

####
prompt_relevance = """
Based on the following evidences:
{contents}

Improve this Argument Pyramid:
{pyramid}

Your task is to enhance the relevance between the claim and each of the arguments. Examine and strengthen the connections to ensure each argument is robustly supported by the evidence provided. The goal is to improve the logical flow and ensure that each argument directly relates to and supports the claim, making the entire pyramid more coherent and closely aligned with the given evidences.

1. **Claim:** Provide a claim or thesis statement that directly answers or addresses the question, ensuring it is in line with how the question is posed (e.g., using 'should' if the question does).

2. **Arguments:** Provide at least five complete sentences as arguments supporting the claim, and include evidence for each argument. Ensure each argument is clearly stated and distinct.

3. **Evidence:** For each argument, provide at least four pieces of evidence to support it. The evidence should be credible, varied (e.g., statistics, expert opinions, real-world examples), and directly relevant to the argument.

Ensure that the structure follows this format:

        - **Claim:** [Your claim here]
          - **Argument 1: [Argument]**
            - Evidence 1: [Detail of the evidence]
            - Evidence 2: [Detail of the evidence]
            - Evidence 3: [Detail of the evidence]
            - Evidence 4: [Detail of the evidence]
          - **Argument 2: [Another Argument]**
            - Evidence 1: [Detail of the evidence]
            - Evidence 2: [Detail of the evidence]
            - Evidence ...
          - **Argument 3**
            - Evidence 1
            - Evidence 2
            - Evidence ...
          - **Argument ...**
            - Evidence 1
            - Evidence 2
            - Evidence ...
"""


####
prompt_support = """
Based on the following evidences:
{contents}

Improve this Argument Pyramid:
{pyramid}

Your task is to enhance the support between each argument and its corresponding evidences. Examine the connections and assess how well the evidence underpins each argument. The goal is to ensure that each argument is strongly backed by the evidence provided, enhancing the overall persuasive power and credibility of the pyramid.

1. **Claim:** Provide a claim or thesis statement that directly answers or addresses the question, ensuring it is in line with how the question is posed (e.g., using 'should' if the question does).

2. **Arguments:** Provide at least five complete sentences as arguments supporting the claim, and include evidence for each argument. Ensure each argument is clearly stated and distinct.

3. **Evidence:** For each argument, provide at least four pieces of evidence to support it. The evidence should be credible, varied (e.g., statistics, expert opinions, real-world examples), and directly relevant to the argument.

Ensure that the structure follows this format:

- **Claim:** [Your claim here]
  - **Argument 1: [Argument]**
    - Evidence 1: [Detail of the evidence]
    - Evidence 2: [Detail of the evidence]
    - Evidence 3: [Detail of the evidence]
    - Evidence 4: [Detail of the evidence]
  - **Argument 2: [Another Argument]**
    - Evidence 1: [Detail of the evidence]
    - Evidence 2: [Detail of the evidence]
    - Evidence ...
  - **Argument 3**
    - Evidence 1
    - Evidence 2
    - Evidence ...
  - **Argument ...**
    - Evidence 1
    - Evidence 2
    - Evidence ...

This modification focuses specifically on enhancing how well the evidence supports the arguments, ensuring that each piece of evidence is not only relevant but effectively strengthens the argument it supports.
"""

####
prompt_coherence = """
Based on the following evidences:
{contents}

Improve this Argument Pyramid:
{pyramid}

Your task is to enhance the logical coherence of the arguments within the pyramid. Assess how each argument logically connects to and supports the main claim. The goal is to ensure that all arguments work together in a coherent and logically consistent manner, collectively strengthening the main claim.

1. **Claim:** Provide a claim or thesis statement that directly answers or addresses the question, ensuring it is in line with how the question is posed (e.g., using 'should' if the question does).

2. **Arguments:** Provide at least five complete sentences as arguments supporting the claim. Ensure each argument is clearly stated and distinct, and logically follows from the claim and from each other.

3. **Evidence:** For each argument, provide at least four pieces of evidence. The evidence should be credible, varied (e.g., statistics, expert opinions, real-world examples), and logically relevant to the argument it supports.

Ensure that the structure follows this format:

- **Claim:** [Your claim here]
  - **Argument 1: [Argument]**
    - Evidence 1: [Detail of the evidence]
    - Evidence 2: [Detail of the evidence]
    - Evidence 3: [Detail of the evidence]
    - Evidence 4: [Detail of the evidence]
  - **Argument 2: [Another Argument]**
    - Evidence 1: [Detail of the evidence]
    - Evidence 2: [Detail of the evidence]
    - Evidence ...
  - **Argument 3**
    - Evidence 1
    - Evidence 2
    - Evidence ...
  - **Argument ...**
    - Evidence 1
    - Evidence 2
    - Evidence ...

This modified prompt focuses on assessing and improving how each argument not only supports the main claim but also aligns and interconnects with other arguments, forming a seamless logical progression that enhances the overall persuasiveness of the pyramid.
"""
####
prompt_completeness = """
Based on the following evidences:
{contents}

Improve this Argument Pyramid:
{pyramid}

Your task is to enhance the completeness of the argument pyramid. Review the structure to ensure it covers all relevant aspects of the issue at hand. Assess whether the pyramid adequately addresses counterarguments and includes a diverse range of evidence. The goal is to ensure that no significant aspect related to the claim is overlooked and that each argument is well-rounded and fully developed.

1. **Claim:** Provide a claim or thesis statement that directly answers or addresses the question, ensuring it is comprehensive and fully reflective of the issue.

2. **Arguments:** Provide at least five complete sentences as arguments supporting the claim. Ensure each argument is clearly stated, distinct, and encompasses a wide range of perspectives and evidence.

3. **Evidence:** For each argument, provide at least four pieces of evidence. The evidence should be credible, varied (e.g., statistics, expert opinions, real-world examples), and cover different dimensions relevant to the argument.

Ensure that the structure follows this format:

- **Claim:** [Your claim here]
  - **Argument 1: [Argument]**
    - Evidence 1: [Detail of the evidence]
    - Evidence 2: [Detail of the evidence]
    - Evidence 3: [Detail of the evidence]
    - Evidence 4: [Detail of the evidence]
  - **Argument 2: [Another Argument]**
    - Evidence 1: [Detail of the evidence]
    - Evidence 2: [Detail of the evidence]
    - Evidence ...
  - **Argument 3**
    - Evidence 1
    - Evidence 2
    - Evidence ...
  - **Argument ...**
    - Evidence 1
    - Evidence 2
    - Evidence ...

This prompt encourages you to ensure that the pyramid not only supports the claim but also fully explores the topic, addresses potential objections, and includes a comprehensive range of supporting details. This will make the entire pyramid more robust and complete, effectively strengthening its persuasive power and validity.
"""

def generate_alternative_pyramid(question, contents, openai_api_key, existing_pyramid):
    """Generate an alternative argument pyramid with a different or opposite claim"""
    openai.api_key = openai_api_key

    # Converts a list of contents into a single string, each separated by two newlines
    formatted_contents = "\n\n".join(contents)

    # Define a prompt that explicitly asks for a different or opposite perspective
    messages = [
        {"role": "system", "content": "You are a skilled debater and expert in constructing logical arguments that consider various perspectives."},
        {"role": "user", "content": f"""
        Based on the following evidences: "{formatted_contents}"

        We previously discussed this claim in our argument pyramid:
        "{existing_pyramid}"

        Now, create a new argument pyramid with a claim that presents a completely different or opposite perspective to the original. Ensure the new claim that directly answers or addresses the question: "{question}".

        1. **Claim:** Propose a new claim that contrasts with or opposes the original.
        2. **Arguments:** Provide at least five complete sentences as arguments supporting the new claim, and include evidence for each argument. Ensure each argument is clearly distinct from those in the original pyramid.
        3. **Evidence:** For each new argument, provide at least four pieces of evidence. The evidence should be credible, varied (e.g., statistics, expert opinions, real-world examples), and support the new, contrasting arguments.

        Ensure that the structure follows this format:

        - **Claim:** [Your new claim here]
          - **Argument 1: [New Argument]**
            - Evidence 1: [Detail of the evidence]
            - Evidence 2: [Detail of the evidence]
            - Evidence 3: [Detail of the evidence]
            - Evidence 4: [Detail of the evidence]
          - **Argument 2: [Another New Argument]**
            - Evidence 1: [Detail of the evidence]
            - Evidence 2: [Detail of the evidence]
            - Evidence ...
          - **Argument 3**
            - Evidence 1
            - Evidence 2
            - Evidence ...
          - **Argument ...**
            - Evidence 1
            - Evidence 2
            - Evidence ...

        Use clear and logical reasoning to establish a compelling alternative perspective.
        """}
    ]

    # Send request to OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        #model="gpt-4",
        messages=messages,
        max_tokens=1500,
        n=1,
        temperature=0.7,
    )

    # Extracting response content
    result = response.choices[0].message['content'].strip()

    return result


def generate_multiple_pyramids(question, threshold, num_pyramids):
    documents = extract_evidence_from_web(question, bing_api_key)
    best_pyramid = None
    best_score = -float('inf')
    existing_pyramids = []

    for i in range(num_pyramids):
        #print(f"Generating pyramid {i+1}/{num_pyramids}...")
        print(f"\nGenerating pyramid {i+1}/{num_pyramids}...\n")
        if i == 0:
            # Use the standard generation method for the first pyramid
            pyramid, total_score = generate_and_evaluate(question, documents, threshold)
        else:
            # Use the alternative generation method for subsequent pyramids
            pyramid, total_score = generate_and_evaluate_alter(question, documents, openai.api_key, existing_pyramids)

        if pyramid:
            #print(pyramid)
            existing_pyramids.append(pyramid)
            if total_score > best_score:
                best_score = total_score
                best_pyramid = pyramid
                #print(f"New best pyramid with score {total_score}:\n{pyramid}\n")

    if best_pyramid:
        print("\n=======================================================")
        print(f"Best pyramid with total score {best_score}:\n{best_pyramid}")
        print("=======================================================")
    else:
        print("Failed to generate a satisfactory pyramid.")

    return best_pyramid

def generate_and_evaluate(question, documents, threshold):
    attempt = 0
    while True:
        pyramid = generate_argument_pyramid(question, documents, openai.api_key)
        print(pyramid)

        relevance_score, support_score, coh_score, completeness_score = calculate(pyramid)
        scores = {
            'relevance': relevance_score,
            'support': support_score,
            'coherence': coh_score,
            'completeness': completeness_score
        }
        min_score_category, min_score = min(scores.items(), key=lambda item: item[1])
        total_score = sum(scores.values())

        if min_score >= threshold:
            print("--------------------------------------------------------")
            print(f"Successful pyramid generated with total score {total_score}")
            print(pyramid)
            print("--------------------------------------------------------")
            return pyramid, total_score
        elif min_score < 0.6:
            attempt += 1
            print(f"Attempt {attempt}: Minimum score of {min_score_category} is below 0.6, re-generating the pyramid.")
            continue  # Re-generate the pyramid if the score is too low
        else:
            attempt += 1
            print(f"Attempt {attempt}: Improving {min_score_category} (current score: {min_score}).")
            prompt = select_prompt(min_score_category)
            pyramid = improve_argument_pyramid(pyramid, documents, prompt, openai.api_key)

def generate_and_evaluate_alter(question, documents, openai_api_key, existing_pyramid):
    attempt = 0
    while True:
        pyramid = generate_alternative_pyramid(question, documents, openai_api_key, existing_pyramid)
        print(pyramid)

        relevance_score, support_score, coh_score, completeness_score = calculate(pyramid)
        scores = {
            'relevance': relevance_score,
            'support': support_score,
            'coherence': coh_score,
            'completeness': completeness_score
        }
        min_score_category, min_score = min(scores.items(), key=lambda item: item[1])
        total_score = sum(scores.values())

        if min_score >= threshold:
            print("--------------------------------------------------------")
            print(f"Successful pyramid generated with total score {total_score}")
            print(pyramid)
            print("--------------------------------------------------------")
            return pyramid, total_score
        elif min_score < 0.6:
            attempt += 1
            print(f"Attempt {attempt}: Minimum score {min_score_category} is below 0.6, re-generating the pyramid.")
            continue  # Re-generate the pyramid if the score is too low
        else:
            attempt += 1
            print(f"Attempt {attempt}: Improving {min_score_category} (current score: {min_score}).")
            prompt = select_prompt(min_score_category)
            pyramid = improve_argument_pyramid(pyramid, documents, prompt, openai.api_key)

def select_prompt(score_category):
    if score_category == 'relevance':
        return prompt_relevance
    elif score_category == 'support':
        return prompt_support
    elif score_category == 'coherence':
        return prompt_coherence
    elif score_category == 'completeness':
        return prompt_completeness
