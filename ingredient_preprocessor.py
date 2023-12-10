from nltk.tokenize import RegexpTokenizer
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
import string
from nltk.stem import WordNetLemmatizer


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
                No need to perform/implement multi-word expression recognition for HW3.
        """
        self.lowercase = lowercase
        self.multiword_expressions = multiword_expressions
        # TODO: Save arguments that are needed as fields of this class

    def find_and_replace_mwes(self, input_tokens: list[str]) -> list[str]:
        """
        IGNORE THIS PART; NO NEED TO IMPLEMENT THIS SINCE NO MULTI-WORD EXPRESSION PROCESSING IS TO BE USED.
        For the given sequence of tokens, finds any recognized multi-word expressions in the sequence
        and replaces that subsequence with a single token containing the multi-word expression.

        Args:
            input_tokens: A list of tokens

        Returns:
            A list of tokens containing processed multi-word expressions
        """
        # NOTE: You shouldn't implement this in homework 
        raise NotImplemented("MWE is not supported")
    
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
        lemmatizer = WordNetLemmatizer()
        
        if self.lowercase:
          
            translator = str.maketrans("", "", string.punctuation)
            # input_tokens_lower = [token.translate(translator) for token in input_tokens]
            new_input_tokens = []
            for token in input_tokens:
                # token = lemmatizer.lemmatize(token)
                token = token.translate(translator)
                token = token.lower().strip()
                split_token = token.split()
                split_toke_list = []
                for toke in split_token:

                    if toke != '':
                        w = lemmatizer.lemmatize(toke, pos='n')

                        split_toke_list.append(w)
                token = ' '.join(split_toke_list)
                new_input_tokens.append(token)

            # input_tokens_lower = [token.lower().strip() for token in input_tokens]
            # input_tokens_lower = [token.translate(translator) for token in input_tokens_lower]
            return new_input_tokens
            return input_tokens_lower
        else:
            
            return [token.lower().strip() for token in input_tokens]
    
    def tokenize(self, text: str) -> list[str]:
        """
        Splits a string into a list of tokens and performs all required postprocessing steps.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        raise NotImplementedError('tokenize() is not implemented in the base class; please use a subclass')
    
class SplitTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses the split function to tokenize a given string.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition 
        """
        super().__init__(lowercase, multiword_expressions)
      

    def tokenize(self, text: str) -> list[str]:
        # TODO: Implement a tokenizer that uses the split function.
        """Split a string into a list of tokens using commas as a delimiter.

        Parameters:

        text [str]: This is an input text you want to tokenize.
        maybe put an int in as a flag to flag instead of a string
        """
        tokens = text.split(sep=',')
        tokens = self.postprocess(tokens)
        return tokens





# TODO (HW3): Take in a doc2query model and generate queries from a piece of text
# Note: This is just to check you can use the models;
#       for downstream tasks such as index augmentation with the queries, use doc2query.csv
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
         
        OPTIONAL (DO NOT DO THIS before you finish the assignment):
            Neural models are best performing when batched to the GPU.
            Try writing a separate function which can deal with batches of documents.
        
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

