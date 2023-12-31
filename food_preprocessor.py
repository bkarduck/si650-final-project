from nltk.tokenize import RegexpTokenizer
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


class Tokenizer:
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        A generic class for objects that turn strings into sequences of tokens.
        A tokenizer can support different preprocessing options or use different methods
        for determining word breaks.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
             
        """
        self.lowercase = lowercase
        self.multiword_expressions = multiword_expressions
        # TODO: Save arguments that are needed as fields of this class

    
    def postprocess(self, input_tokens: list[str]) -> list[str]:
        """
        Performs any set of optional operations to modify the tokenized list of words such as
        lower-casing and returns the modified list of tokens.

        Args:
            input_tokens: A list of tokens

        Returns:
            A list of tokens processed by lower-casing depending on the given condition
        """
        # TODO: Add support for lower-casing
        if self.lowercase:
            input_tokens_lower = [token.lower() for token in input_tokens]
            return input_tokens_lower
        else:
            return input_tokens
    
    def tokenize(self, text: str) -> list[str]:
        """
        Splits a string into a list of tokens and performs all required postprocessing steps.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        raise NotImplementedError('tokenize() is not implemented in the base class; please use a subclass')


class RegexTokenizer(Tokenizer):
    def __init__(self, token_regex: str, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses NLTK's RegexpTokenizer to tokenize a given string.

        Args:
            token_regex: Use the following default regular expression pattern: '\\w+'
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
               
        """
        super().__init__(lowercase, multiword_expressions)
        # TODO: Save a new argument that is needed as a field of this class
        self.token_regex = token_regex
        # TODO: Initialize the NLTK's RegexpTokenizer 
        self.tokenizer = RegexpTokenizer(self.token_regex)

    def tokenize(self, text: str) -> list[str]:
        """Uses NLTK's RegexTokenizer and a regular expression pattern to tokenize a string.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # TODO: Tokenize the given text and perform postprocessing on the list of tokens using the postprocess function
        tokens = self.tokenizer.tokenize(text)
        tokens = self.postprocess(tokens)
        return tokens


# TODO Take in a doc2query model and generate queries from a piece of text
# Note: This is just to check you can use the models; for downstream tasks such as index augmentation with the queries, use doc2query.csv
class Doc2QueryAugmenter:
    """
    This class is responsible for generating queries for a document.
    These queries can augment the document before indexing.

    MUST READ: https://huggingface.co/doc2query/msmarco-t5-base-v1

    OPTIONAL reading
        1. Document Expansion by Query Prediction (Nogueira et al.): https://arxiv.org/pdf/1904.08375.pdf
    """
    def __init__(self, doc2query_model_name: str = 'doc2query/msmarco-t5-base-v1') -> None:
        """
        Creates the T5 model object and the corresponding dense tokenizer.
        
        Args:
            doc2query_model_name: The name of the T5 model architecture used for generating queries
        """
        self.device = torch.device('cpu')  # Do not change this unless you know what you are doing

        # TODO: Create the dense tokenizer and query generation model using HuggingFace transformers
        self.tokenizer = T5Tokenizer.from_pretrained(doc2query_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(doc2query_model_name)

    def get_queries(self, document: str, n_queries: int = 5, prefix_prompt: str = '') -> list[str]:
        """
        Steps
            1. Use the dense tokenizer/encoder to create the dense document vector.
            2. Use the T5 model to generate the dense query vectors (you should have a list of vectors).
            3. Decode the query vector using the tokenizer/decode to get the appropriate queries.
            4. Return the queries.
         
            Ensure you take care of edge cases.
         
        
        Args:
            document: The text from which queries are to be generated
            n_queries: The total number of queries to be generated
            prefix_prompt: An optional parameter
                Some models are not fine-tuned to generate queries.
                So we need to add a prompt to coax the model into generating queries.
                This string enables us to create a prefixed prompt to generate queries for the models.
                Prompt-engineering: https://en.wikipedia.org/wiki/Prompt_engineering
        
        Returns:
            A list of query strings generated from the text
        """
        # Note: Feel free to change these values to experiment
        document_max_token_length = 400  # as used in OPTIONAL Reading 1
        top_p = 0.85

        # TODO: For the given model, generate a list of queries that might reasonably be issued to search
        #       for that document
        if len(document) > document_max_token_length:
            document = document[:document_max_token_length]
        if prefix_prompt != '':
            document = prefix_prompt + document
        if n_queries == 0:
            return []
        input_ids = self.tokenizer.encode(document, return_tensors='pt')
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=64,
            do_sample=True,
            top_p=top_p,
            num_return_sequences=n_queries
        )
        queries = []
        for i in range(len(outputs)):
            queries.append(self.tokenizer.decode(outputs[i], skip_special_tokens=True))
        # NOTE: Do not forget edge cases
        return queries


# Don't forget that you can have a main function here to test anything in the file

