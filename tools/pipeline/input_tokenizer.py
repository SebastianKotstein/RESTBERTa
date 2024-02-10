'''
Copyright 2023 Sebastian Kotstein

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''

from transformers import AutoTokenizer
import numpy as np

class InputTokenizer: 

    def __init__(self, base_model: str, max_length: int = 512, doc_stride: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        #self.tokenizer.save_pretrained("/home/user/2023_02_16_QA/checkpoints")
        self.max_length = max_length
        self.doc_stride = doc_stride


    def mask_offset_mapping(self, sequence_ids, offset_mapping):
        """
        Masks the passed 'offset_mapping', i.e., replaces all entries with 'None' that represent tokens that are not part of the context.
        The method returns the masked offset mapping.
        
        Parameters
        ----------
        sequence_ids :
            Vector that defines for each token whether the token is part of the query, of the paragraph, or is a special token.
            An entry representing a token that is part of the paragraph must have the value '1'.
        offset_mapping
            Vector that contains for each token its start and end index on character level in the original input (consisting of question, context, and special characters). 
        
        Returns
        -------
        Masked 'offset_mapping'; entries that represent tokens that are not part of the context have the value 'None'
        """
        masked_offset_mapping = [(o if sequence_ids[k] == 1 else None) for k, o in enumerate(offset_mapping)]
        return masked_offset_mapping
    
    def extract_fragment(self, sequence_ids, input_ids):
        """
        Extracts the fragment from the input sequence and returns the list of tokens belonging to the fragment and the fragment itself.

        Parameters
        ----------
        sequence_ids:  
          Vector that defines for each token whether the token is part of the query, of the paragraph, or is a special token.
          An entry representing a token that is part of the paragraph must have the value '1'
        input_ids:
          Vector of input indices, i.e., tokens in their numerical representation

        Returns
        -------
        List of tokens, i.e., [string], belonging to the fragment and the fragment sequence as string
        """
        # create positive mask for indices belonging to the fragment
        mask = np.array(sequence_ids, dtype=bool)
        # transform input indices into numpy array
        np_input_ids = np.array(input_ids)
        # convert input indices belonging to the fragment into their token representation, i.e., list of strings
        tokens_of_fragments = self.tokenizer.convert_ids_to_tokens(np_input_ids[mask])
        # convert, i.e., decode, input indices belonging to the fragment into sentence (string)
        fragment = self.tokenizer.decode(np_input_ids[mask])

        return tokens_of_fragments, fragment


    def tokenize(self, batch):

        # Tokenizes the QA samples of the passed batch. Each QA sample may result into multiple tokenized samples if the input sequence, consisting of query and paragraph, exceeds the model's input size (typically 512 tokens). 
        # If a QA sample must be split into multiple tokenized samples, only the paragraph will be split by the tokenizer so that every resulting tokenized sample will contain the original query plus another fragment of the original paragraph. 
        # Note that the resulting fragments overlap by the number of tokens specifiec in 'doc_stride'. Example: If a 'doc_stride' of 128 is set, the second fragment will start with the last 128 tokens of the first fragment, and so further.
        tokenized_samples = self.tokenizer(
            batch["qa_sample_query"],
            batch["qa_sample_paragraph"],
            truncation="only_second",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # ID of the sample (string), e.g. "7fed77b9abe24a2db869c8b9919a1e9b"
        tokenized_samples["qa_sample_id"] = []
        # Title of the sample (string), e.g. "my very urgent question"
        tokenized_samples["qa_sample_title"] = []
        # Query, i.e., query, of the sample (string), e.g. "The name of a user"
        tokenized_samples["qa_sample_query"] = []

        # ID of the paragraph (string), e.g. "7fed77b9abe24a2db869c8b9919a1e9b"
        tokenized_samples["qa_sample_paragraph_id"] = []
        # Title of the paragraph (string), e.g. "schema XYZ"
        tokenized_samples["qa_sample_paragraph_title"] = []
        # Paragraph (context) of the sample (string), e.g. "users[*].id users[*].name _links.href _links.rel"
        tokenized_samples["qa_sample_paragraph"] = []

        # fragment of the tokenized sample
        tokenized_samples["tokenized_sample_fragment"] = []
        # input tokens of the tokenized sample ([string])
        tokenized_samples["tokenized_sample_tokens"] = []
        # input tokens of the fragment of the tokenized sample ([string])
        tokenized_samples["tokenized_sample_fragment_tokens"] = []

        # Index of the CLS token (int)
        tokenized_samples["tokenized_sample_cls_index"] = []

        # List of indices that maps a tokenized sample (index position) to the QA sample (index value) it results from. 
        # Example: The list [0,0,1,2,2] states that the first two tokenized samples belong to the first QA sample, while the third tokenized sample had resulted from the second QA sample. The fourth and fifth tokenized samples belong to the third QA sample.
        sample_mapping = tokenized_samples.pop("overflow_to_sample_mapping")
        
        # Iterate over all tokenized samples and extract its offset_mapping
        for i, offset_mapping in enumerate(tokenized_samples["offset_mapping"]):
            
            # Mask offset mapping
            masked_offset_mapping = self.mask_offset_mapping(
                sequence_ids = tokenized_samples.sequence_ids(i),
                offset_mapping = offset_mapping)

            # Load the index of the original QA-sample
            sample_index = sample_mapping[i]

            # Load cls_index
            cls_index = tokenized_samples["input_ids"][i].index(self.tokenizer.cls_token_id)
    
            # overwrite offset_mapping
            tokenized_samples["offset_mapping"][i] = masked_offset_mapping

            # append ID of the QA sample
            tokenized_samples["qa_sample_id"].append(batch["qa_sample_id"][sample_index])
            # append title of the QA sample
            tokenized_samples["qa_sample_title"].append(batch["qa_sample_title"][sample_index])
            # append query of the QA sample
            tokenized_samples["qa_sample_query"].append(batch["qa_sample_query"][sample_index])

            # append ID of the paragraph of the QA sample
            tokenized_samples["qa_sample_paragraph_id"].append(batch["qa_sample_paragraph_id"][sample_index])
            # append title of the paragraph of the QA sample
            tokenized_samples["qa_sample_paragraph_title"].append(batch["qa_sample_paragraph_title"][sample_index])
            # append paragraph of the QA sample
            tokenized_samples["qa_sample_paragraph"].append(batch["qa_sample_paragraph"][sample_index])

            if batch["verbose_output"][sample_index]:
                
                # extract fragment from input sequence
                tokens_of_fragments, fragment = self.extract_fragment(tokenized_samples.sequence_ids(i),tokenized_samples["input_ids"][i])
                # convert all input indices into their token representation, i.e., list of strings
                tokens = self.tokenizer.convert_ids_to_tokens(tokenized_samples["input_ids"][i])

                # append fragment of the tokenized sample
                tokenized_samples["tokenized_sample_fragment"].append(fragment)
                # append input tokens of the tokenized sample 
                tokenized_samples["tokenized_sample_tokens"].append(tokens)
                # append input tokens of the fragment of the tokenized sample
                tokenized_samples["tokenized_sample_fragment_tokens"].append(tokens_of_fragments)
            else:
                tokenized_samples["tokenized_sample_fragment"].append(None)
                tokenized_samples["tokenized_sample_tokens"].append(None)
                tokenized_samples["tokenized_sample_fragment_tokens"].append(None)

            # append cls_index
            tokenized_samples["tokenized_sample_cls_index"].append(cls_index)     
              
            
        return tokenized_samples


        







