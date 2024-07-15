import torch, openai
import numpy as np
from utilities.helpers import parse_response, safe_average

from transformers import BertTokenizer, AutoModelForSequenceClassification, RobertaTokenizer, AlbertTokenizer, AlbertForSequenceClassification
from models.roberta_model.Roberta import ContrastiveRoberta
from models.roberta_model.Roberta import SupportScoreModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer1 = BertTokenizer.from_pretrained('bert-base-uncased')
model_path1 = 'models/bert_model/'
model1 = AutoModelForSequenceClassification.from_pretrained(model_path1)
model1 = model1.to(device)

# Load Model 2: Contrastive Roberta
tokenizer2 = RobertaTokenizer.from_pretrained('roberta-base')
contrastive_model = ContrastiveRoberta()
model_path2 = 'models/roberta_model/best_model.pt'
contrastive_model.load_state_dict(torch.load(model_path2, map_location=device))
support_model = SupportScoreModel(contrastive_model)
support_model = support_model.to(device)

# Load Model 3: Albert-based classifier
tokenizer3 = AlbertTokenizer.from_pretrained('models/albert_model/')
model3 = AlbertForSequenceClassification.from_pretrained('models/albert_model/')
model3 = model3.to(device)
model3.eval()

def calculate(argument_pyramid):
    """
        Calculate and evaluate different metrics of an argument pyramid to determine its effectiveness.
        This includes calculating relevance, support, coherence, and completeness scores.

        Args:
        argument_pyramid (str): A string that includes the entire argument pyramid structure as provided by GPT.

        Returns:
        tuple: Returns a tuple containing scores for relevance, support, coherence, and completeness.
        """
    parsed_response = parse_response(argument_pyramid)
    claim = parsed_response['claim']
    arguments = parsed_response['arguments']

    relevance_scores = calculate_claim_argument_quality(claim, arguments, tokenizer1, model1, device)
    print(f"\nRelevance_scores: {relevance_scores}")

    support_scores = calculate_argument_evidence_quality(arguments, tokenizer2, support_model, device)
    print(f"Support_scores: {support_scores}")

    global_score, local_scores = calculate_coh_scores(claim, arguments, tokenizer3, model3, device)
    print("Global Coherence Score:", global_score)
    print("Local Coherence Scores:", local_scores)

    relevance_score = safe_average(relevance_scores)
    support_score = np.nanmean(support_scores)

    lambda_val = 0.2

    num_local_scores = len(local_scores)
    weights = [0.4] + [(0.6 / (num_local_scores - 1)) for _ in range(num_local_scores - 1)]
    weighted_local_score = np.average(local_scores, weights=weights)
    coh_score = global_score * (1 - lambda_val) + weighted_local_score * lambda_val

     # Generate questions
    questions = generate_questions(claim)

    # Check answers
    compl_scores = check_complet(questions, arguments)
    completeness_score = sum(compl_scores) / len(questions) if questions else 0
    completeness_score = np.nanmean(np.nan_to_num(compl_scores))


    # Print each question and its score
    for question, score in zip(questions, compl_scores):
        print(f"Question: {question}\nScore: {score}\n")

    print(f"""
    +-------------------------------------+
    | Overall completeness score: {completeness_score:.3f}
    | Relevance score: {relevance_score:.3f}
    | Support score: {support_score:.3f}
    | Coherence Score: {coh_score:.3f}
    +-------------------------------------+
    """)
    #total_score = alpha * relevance_score + beta * support_score + gamma * coh_score + completeness_score
    #print("Total Argument—Pyramid Score:", total_score)
    return relevance_score, support_score, coh_score, completeness_score



def calculate_claim_argument_quality(claim, arguments, tokenizer1, model1, device):
    relevance_scores = []
    for argument in arguments:
        input_text = claim + " [SEP] " + argument['title']
        inputs = tokenizer1(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model1(**inputs)
            logits = outputs.logits
            logits = torch.clamp(logits, min=0, max=1)
            predicted_quality_score = logits.squeeze().item()
            relevance_scores.append(predicted_quality_score)

    return relevance_scores

def calculate_argument_evidence_quality(arguments, tokenizer2, support_model, device):
    support_scores = []
    for argument in arguments:
        evidence_scores = []
        for evidence in argument['evidence']:
            arg_tokens = tokenizer2(argument['title'], return_tensors='pt', padding="max_length", truncation=True, max_length=512)
            evidence_tokens = tokenizer2(evidence, return_tensors='pt', padding="max_length", truncation=True, max_length=512)
            arg_tokens = {k: v.to(device) for k, v in arg_tokens.items()}
            evidence_tokens = {k: v.to(device) for k, v in evidence_tokens.items()}

            with torch.no_grad():
                support_score = support_model(arg_tokens, evidence_tokens)
                evidence_scores.append(support_score.item())
        support_scores.append(evidence_scores)

    return support_scores


def calculate_coh_scores(claim, arguments, tokenizer3, model3, device):
    text_blocks = []

    # First block: claim, followed by a combination of each argument and all its evidences
    first_block = [claim]
    for arg in arguments:
        argument_title = arg['title']
        evidences_text = " ".join(arg['evidence'])
        first_block.append(argument_title + " " + evidences_text)
    text_blocks.append(" ".join(first_block))

    # Second block: claim and all arguments title combinations
    all_arguments_titles = " ".join([arg['title'] for arg in arguments])
    text_blocks.append(claim + " " + all_arguments_titles)

    # Subsequent blocks: Each argument and all its evidences form a separate block.
    for arg in arguments:
        argument_title = arg['title']
        evidences_text = " ".join(arg['evidence'])
        text_blocks.append(argument_title + " " + evidences_text)

    # Encode and compute scores
    encoding = tokenizer3(text_blocks, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model3(**encoding)
        logits = torch.softmax(outputs.logits, dim=-1)

        # Global score: the score of the first block
        global_score = logits[0][1].item()
        # Local scores: scores of the remaining blocks
        local_scores = [logits[i][1].item() for i in range(1, len(logits))]

    return global_score, local_scores


def generate_questions(claim):
    # Construct a prompt that encourages the generation of comprehensive questions
    prompt = f"""
    Considering the claim '{claim}', generate a list of detailed questions that cover various critical aspects necessary to assess the validity and completeness of supporting arguments. Focus on areas such as:
    - General Understanding
    - Context and Relevance
    - Assumptions and Implications
    - Counterarguments and Criticisms
    - Specific Aspects and Details
    Each question should invite a deep analysis of the arguments.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        #model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )
    questions = response['choices'][0]['message']['content'].strip().split('\n')
    return [q.strip() for q in questions if q.strip() != '']


def check_complet(questions, arguments):
    scores = []
    arguments_text = " ".join([arg['title'] + " " + " ".join(arg['evidence']) for arg in arguments])

    for question in questions:
        prompt = f"""
        Assess the response based on how well the provided arguments address this question:
        '{question}'
        Arguments: {arguments_text}
        Score the response as follows:
        - 1.0 for fully addressed and well supported
        - 0.75 for mostly addressed but some details missing
        - 0.5 for adequately addressed but lacking depth
        - 0.25 for minimally addressed with very little detail
        - 0.0 for not addressed at all
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            #model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}]
        )
        answer = response['choices'][0]['message']['content'].lower()

        if "1.0" in answer:
            scores.append(1.0)
        elif "0.75" in answer:
            scores.append(0.75)
        elif "0.5" in answer:
            scores.append(0.5)
        elif "0.25" in answer:
            scores.append(0.25)
        else:
            scores.append(0.0)

    return scores