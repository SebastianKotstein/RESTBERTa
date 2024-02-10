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

import numpy as np

class OutputInterpreter:
    
    def __init__(self, best_size: int) -> None:
        self.n_best_size = best_size
    
    def interpret_output(self, tokenized_samples, model_output, batch_size, suppress_duplicates = False):
        
        # TODO: Convert score to str in a final step, i.e., after processing results (instead of converting it temporarily back to float for sorting)
        results = self.create_empty_results_dict()

        for i in range(batch_size):
            # create and append result entry to the list of results
            if tokenized_samples["qa_sample_id"][i] not in results["qa_sample_id"]:
                
                # append ID of the QA sample to the list of results
                results["qa_sample_id"].append(tokenized_samples["qa_sample_id"][i])
                # append title of the QA sample to the list of results
                results["qa_sample_title"].append(tokenized_samples["qa_sample_title"][i])
                # append query of the QA sample to the list of results
                results["qa_sample_query"].append(tokenized_samples["qa_sample_query"][i])

                # append ID of the paragraph of the QA sample to the list of results
                results["qa_sample_paragraph_id"].append(tokenized_samples["qa_sample_paragraph_id"][i])
                # append title of the paragraph of the QA sample to the list of results
                results["qa_sample_paragraph_title"].append(tokenized_samples["qa_sample_paragraph_title"][i])
                # append paragraph of the QA sample to the list of results
                results["qa_sample_paragraph"].append(tokenized_samples["qa_sample_paragraph"][i])

                # append empty list of tokenized samples to the list of results
                results["tokenized_samples"].append([])

            # query index of result entry 
            index = list(results["qa_sample_id"]).index(tokenized_samples["qa_sample_id"][i])    

            # prepare tokenized sample object
            tokenized_sample = {
                "tokens": tokenized_samples["tokenized_sample_tokens"][i],
                "fragment": tokenized_samples["tokenized_sample_fragment"][i],
                "fragment_tokens": tokenized_samples["tokenized_sample_fragment_tokens"][i]
            }

            # interpret prediction
            answers = self.get_answers(
                offset_mapping=tokenized_samples["offset_mapping"][i],
                cls_index=tokenized_samples["tokenized_sample_cls_index"][i],
                paragraph=tokenized_samples["qa_sample_paragraph"][i],
                predicted_start_logits=model_output.start_logits[i],
                predicted_end_logits=model_output.start_logits[i],
                suppress_duplicates=suppress_duplicates
            )
            tokenized_sample["answers"] = answers

            # append tokenized sample and its result to the list of results
            results["tokenized_samples"][index].append(tokenized_sample)
        

        for i in range(len(results["qa_sample_id"])):
            # prepare list of combined answers, i.e., answers over all tokenized samples
            combined_answers = []
            # iterate over all tokenized samples of QA sample
            for tokenized_sample in results["tokenized_samples"][i]:
                # iterate over all answers of tokenized sample
                for answer in tokenized_sample["answers"]:
                    # append answer of tokenized sample if it is NOT an out of span answer
                    if answer["property"] is not None:
                        combined_answers.append(answer)
            
            # sort list of combined answers
            combined_answers = sorted(combined_answers, key=lambda x: float(x["score"]), reverse=True)
            results["answers"].append(combined_answers)
        
        return results
    
    def create_empty_results_dict(self):
        results = dict()

        # ID of the sample (string), e.g. "7fed77b9abe24a2db869c8b9919a1e9b"
        results["qa_sample_id"] = []
        # Title of the sample (string), e.g. "my very urgent question"
        results["qa_sample_title"] = []
        # Query, i.e., query, of the sample (string), e.g. "The name of a user"
        results["qa_sample_query"] = []

        # ID of the paragraph (string), e.g. "7fed77b9abe24a2db869c8b9919a1e9b"
        results["qa_sample_paragraph_id"] = []
        # Title of the paragraph (string), e.g. "schema XYZ"
        results["qa_sample_paragraph_title"] = []
        # Paragraph (context) of the sample (string), e.g. "users[*].id users[*].name _links.href _links.rel"
        results["qa_sample_paragraph"] = []

        # ranked list of answers over all tokenized samples
        results["answers"] = []
        # list of tokenized samples and their results
        results["tokenized_samples"] = []

        return results

    def identify_properties(self,context,start_char_index,end_char_index):
        """
        Identifies the properties that are (partially) covered by the span starting at 'start_char_index' and ending at 'end_char_index' in the specified context.
        The method returns the list of properties that are fully or partially covered. 
        Each property contained in this list is a dictionary and has the following structure:
        {
            'name': full property name in XPath style (str)
            'partial_name': concatenated characters of the property name that are covered (str)
            'length': number of characters of the property that are covered (int)
            'partial': flag indicating whether property is fully (False) or partially (True) covered
            'start_char_index': start index of the property on character level in the context (int)
            'end_char_index': end index of the property on character level in the context (int)
        }
        Note: If the property is only partially covered, the fields 'start_char_index' and 'end_char_index' point to the start and end of the partial, not the full property

        Parameters
        ----------
        context : str
            Original context
        start_char_index : int
            Start index of the span on character level in the original context
        end_char_index : int
            End index of the span on character level in the original context

        Returns
        -------
        List of identified properties
        """
        properties = []
        current_property = None
        is_on_property = False

        # iterate over each character of the span
        for i in range(end_char_index-start_char_index):
            index = start_char_index+i
            c = context[index]
            
            # if character is a property separator
            if c == " ":
                # if there is still an unfinished property
                if is_on_property:
                    # finalize property
                    current_property["end_char_index"] = index
                    
                    # if start index of property is start index of whole span (answer), i.e. there are characters belonging to the property that are before span
                    if current_property["start_char_index"] == start_char_index:
                        # go backward in context and determine full property name
                        back_counter = start_char_index-1
                        while back_counter >= 0 and context[back_counter] != " ":
                            current_property["name"] = context[back_counter] + current_property["name"]
                            current_property["partial"] = True # only True (i.e partial), if we have to go backward
                            back_counter-=1
                    properties.append(current_property)
                    current_property = None
                is_on_property = False
            else:
                # if this is the first character of a new property
                if not is_on_property:
                    # prepare a new property
                    current_property = {
                        "name": c,
                        "partial_name": c,
                        "length": 1,
                        "partial": False,
                        "start_char_index": index,
                        "end_char_index": None
                    }
                else:
                    # else, append character to current property
                    current_property["name"]+= c
                    current_property["partial_name"]+= c
                    current_property["length"]+=1
                
                # in both cases (either a new or an existing one), we are on a property 
                is_on_property = True

        # after iterating over all characters of span
        # check whether there is an unfinished property:
        if current_property:
            current_property["end_char_index"] = end_char_index
            
            # go forward in context and determine full property name
            forward_counter = end_char_index
            while forward_counter < len(context) and context[forward_counter] != " ":
                current_property["name"] = current_property["name"] + context[forward_counter]
                current_property["partial"] = True # only True (i.e partial), if we have to go forward
                forward_counter+=1
                
            
            # Special case: If the unfinished property is the only identified property (i.e. properties is empty),
            # we had not the chance to go backward yet (since going backward is only possible, when finalizing a property, see code above)
            # Thus go backward in context and determine full property name
            if len(properties) == 0:
                back_counter = start_char_index-1
                while back_counter >= 0 and context[back_counter] != " ":
                    current_property["name"] = context[back_counter] + current_property["name"]
                    current_property["partial"] = True # only True (i.e partial), if we have to go backward
                    back_counter-=1
                    
            properties.append(current_property)
            
        return properties

    def determine_best_property(self,properties):
        """
        Determines the 'best' property from the passed list of properties. First, the method scans the input list for properties that are fully covered.
        Only if exactly one fully covered property exists, then the method determines and returns this properties as the best property. If two or more
        fully covered properties exist, the method will return 'None' due to this conflict. If the list does not contain any fully covered property,
        the method scans for partial properties in step two: The method determines and returns the partial covered property with the longest sequence of
        covered characters. If multiple partial covered properties share this longest sequence (i.e. having the same length), the method will return 'None'
        due to this conflict. Furthermore, the method returns 'None' if the input list is empty.
        
        Parameters
        ----------
        properties : [dict()]
            List of properties (use identify_properties(...) to identify these properties)
        
        Returns
        -------
        The 'best' property or 'None'
        """
        if not properties:
            # no properties
            return None
        else:
            # first, search for full properties
            full_property = None
            for p in properties:
                if not p["partial"]:
                    if full_property is None:
                        full_property = p
                    else:
                        # at least two full properties (--> conflict)
                        return None
            if full_property:
                # one full property
                return full_property
            else:
                # then, search for partial properties
                partial_property = None
                length_conflict = False
                for p in properties:
                    if p["partial"]:
                        if partial_property is None:
                            partial_property = p
                        else:
                            if partial_property["length"] < p["length"]:
                                partial_property = p
                                length_conflict = False
                            elif partial_property["length"] == p["length"]:
                                length_conflict = True
                
                if length_conflict:
                    # two longest partial properties have same length (--> conflict)
                    return None
                else:
                    return partial_property
                
    def are_indices_out_of_context(self, start_index, end_index, offset_mapping):
        """
        Returns 'True' if the span defined by the passed start and end index (token level) does not completely lies within the context, else 'False'.
        The boundaries of the context (i.e. start and end index) are defined within the passed offset_mapping: 
        Every entry that represents a token that is not part of the context has the value 'None'.
        
        Parameters
        ----------
        start_index : int
            Start index (on token level) of the span
        end_index : int
            End index (on token level) of the span
        offset_mapping
            Vector that contains for each token its start and end index on character level in the original input (consisting of question, context, and special characters). 
            All entries that represent a token that is not part of the context must be 'None' (use the method 'mask_offset_mapping(...)' to mask the 'offset_mapping' vector before using this method).
            
        Returns
        -------
        'True' if the span does not completely lies within the context, else 'False'
        """
        return (start_index >= len(offset_mapping) or end_index >= len(offset_mapping) #if indices are out of bound (should never happen????)
                or offset_mapping[start_index] is None or offset_mapping[end_index] is None)

    def is_end_before_start(self, start_index, end_index):
        """
        Returns 'True' if the passed 'end_index' is smaller than (i.e. before) the passed 'start_index', else 'False'.
        
        Parameters
        ----------
        start_index : int
            Start index (on token level) of the span
        end_index : int
            End index (on token level) of the span
            
        Returns
        -------
        'True' if the passed 'end_index' is smaller than (i.e. before) the passed 'start_index', else 'False'.
        """
        return end_index < start_index

    def is_answer_too_long(self, start_index, end_index,max_length):
        """
        Returns 'True', if the number of tokens in the span defined by the passed start and end index (token level) exceeds the passed length, else 'False'.
        
        Parameters
        ----------
        start_index : int
            Start index (on token level) of the span
        end_index : int
            End index (on token level) of the span
        max_length : int
            Maximum length
            
        Returns
        -------
        'True', if the number of tokens in the span exceeds the passed length, else 'False'.
        """
        if max_length:
            return end_index - start_index + 1 > max_length
        else:
            return False
    

    def get_answers(self, offset_mapping, cls_index, paragraph, predicted_start_logits, predicted_end_logits, suppress_duplicates=False):
        
        # Gather the indices for the best start/end logits (index syntax is: [stop:start:steps] with steps = -1 --> negative order)
        # np.argsort returns a sorted list of indices in ascending order, therefore, we gather the last 'n_best_size' indices
        # in reverse order (syntax: [stop:start:steps] with steps = -1 --> negative order)
        #(see https://towardsdatascience.com/the-basics-of-indexing-and-slicing-python-lists-2d12c90a94cf)
        best_start_indices = np.argsort(predicted_start_logits)[-1 : -self.n_best_size - 1 : -1].tolist()
        best_end_indices = np.argsort(predicted_end_logits)[-1 : -self.n_best_size - 1 : -1].tolist()   
        
        # prepare list for answers
        valid_answers = []
        
        for start_index in best_start_indices:
            for end_index in best_end_indices:
                # Do not consider....
                
                # Case 1:) Answers that are out of context
                # In this case, either start_index or end_index (or both) point to a token positions outside the context
                # Remember: We have set all positions of tokens, which are out of context, with 'None' in "offset_mapping" (see tokenize_validation_samples)
                if self.are_indices_out_of_context(start_index,end_index,offset_mapping):
                    continue
                    
                # Case 2:) Answers where end is before start index
                if self.is_end_before_start(start_index, end_index):
                    continue
                
                # Optional case 3:) Answers that are too long
                #if self.is_answer_too_long(start_index, end_index,self.max_answer_length):
                #    continue
                
                
                start_char_index = offset_mapping[start_index][0]
                end_char_index = offset_mapping[end_index][1]
                
                # identify properties and determine best property
                properties = self.identify_properties(paragraph,start_char_index,end_char_index)
                best_property = self.determine_best_property(properties)
                
                # Case 4:) Answers that do not point clearly to a property (without conflicts)
                if best_property is None:
                    continue
                
                # add answer:
                valid_answers.append(
                    {
                        "score": predicted_start_logits[start_index] + predicted_end_logits[end_index],
                        "span": paragraph[start_char_index:end_char_index],
                        "start_char_index": start_char_index,
                        "end_char_index": end_char_index,
                        "property": best_property
                    }
                )
        
        # finally, add NULL answer as valid answer
        valid_answers.append(
            {
                "score": predicted_start_logits[cls_index] + predicted_end_logits[cls_index],
                "span": None,
                "start_char_index": cls_index,
                "end_char_index": cls_index,
                "property": None
            }
        )
        
        # sort valid answers by score in descending order
        sorted_valid_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)
        
        if suppress_duplicates:
            without_duplicates = {}
            for answer in sorted_valid_answer:
                if answer["property"] is not None and answer["property"]["name"] in without_duplicates: 
                    if answer["score"] > without_duplicates[answer["property"]["name"]]["score"]:
                        without_duplicates[answer["property"]["name"]] = answer
                elif "<no-answer>" not in without_duplicates and answer["property"] is None:
                    without_duplicates["<no-answer>"] = answer
                else:
                    without_duplicates[answer["property"]["name"]] = answer
                    
            sorted_valid_answer = [x for x in without_duplicates.values()]

        scores = np.array([answer["score"] for answer in sorted_valid_answer])
        softmax = np.exp(scores)/sum(np.exp(scores))
        
        for i in range(len(sorted_valid_answer)):
            sorted_valid_answer[i]["score"] = str(sorted_valid_answer[i]["score"])
            sorted_valid_answer[i]["probability"] = str(softmax[i])
            
        return sorted_valid_answer