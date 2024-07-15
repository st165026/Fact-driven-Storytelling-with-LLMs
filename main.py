from generate_argument_pyramid.generate_argument_pyramid import generate_multiple_pyramids
from config.config import threshold
from config.config import openai_api_key
import openai

# Users can choose their own questions and need answers in many aspects.
question = "Where should Disney build its next theme park?"
num_pyramids = 3


def main():
    openai.api_key = openai_api_key
    successful_pyramid = generate_multiple_pyramids(question, threshold, num_pyramids)


if __name__ == "__main__":
    main()
