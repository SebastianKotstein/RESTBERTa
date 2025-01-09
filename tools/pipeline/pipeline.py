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


from .input_tokenizer import InputTokenizer
from .qa_model import QAModel
from .output_interpreter import OutputInterpreter

import uuid

import numpy as np

class InvalidRequestException(Exception):
    def __init__(self, message="Invalid request") -> None:
        self.message = message
        super().__init__(self.message)

class Pipeline:
    
    def __init__(self, model_checkpoint, best_size = 20, cache = None, token = None) -> None:
        self.tokenizer = InputTokenizer("microsoft/codebert-base")
        self.model = QAModel(model_checkpoint, token=token)
        self.interpreter = OutputInterpreter(best_size)
        self.cache = cache
    
    def process(self, input_dict, top = None, suppress_duplicates = False, no_answer_strategy = None):
        input_dict = self.sort_schema_values(input_dict)
        batch = self.json_to_batch(input_dict,no_answer_strategy)
        if len(batch["qa_sample_id"]):
            tokenized_samples = self.tokenizer.tokenize(batch)
            output, batch_size = self.model.predict(tokenized_samples)
            results = self.interpreter.interpret_output(tokenized_samples,output,batch_size,no_answer_strategy)
        else:
            results = self.interpreter.create_empty_results_dict()
        merged_output, to_be_cached = self.merge_results_w_input_json(input_dict,results,no_answer_strategy)
        self.store_items_in_cache(to_be_cached,no_answer_strategy)
        merged_output = self.limit_results(merged_output,top,suppress_duplicates)
        return self.calculate_probabilites(merged_output)
    
    def sort_schema_values(self, input_dict):
        if "schemas" in input_dict:
            for i,schema in enumerate(input_dict["schemas"]):
                if "value" in schema:
                    value = input_dict["schemas"][i]["value"].strip().split()
                    value.sort()
                    input_dict["schemas"][i]["value"] = " ".join(value)
        return input_dict


    def limit_results(self, input_dict, top = None, suppress_duplicates = False):
        for schema in input_dict["schemas"]:
            for query in schema["queries"]:
                if query["result"]:
                    if suppress_duplicates:
                        query["result"]["answers"] = self.suppress_duplicates(query["result"]["answers"])
                        for tokenized_sample in query["result"]["tokenizedSamples"]:
                            tokenized_sample["answers"] = self.suppress_duplicates(tokenized_sample["answers"])
                    if top and len(query["result"]["answers"])>top:
                        query["result"]["answers"] = query["result"]["answers"][0:top]
                        for tokenized_sample in query["result"]["tokenizedSamples"]:
                            if len(tokenized_sample["answers"])>top:
                                tokenized_sample["answers"] = tokenized_sample["answers"][0:top]
        return input_dict
    
    def suppress_duplicates(self, answers):
        without_duplicates = {}
        for answer in answers:
            if answer["property"] is not None:
                if answer["property"]["name"] not in without_duplicates:
                    without_duplicates[answer["property"]["name"]] = answer
            else:
                if "<no-answer>" not in without_duplicates:
                    without_duplicates["<no-answer>"] = answer
        return [x for x in without_duplicates.values()]
    
    def calculate_probabilites(self, input_dict):
        for schema in input_dict["schemas"]:
            for query in schema["queries"]:
                if query["result"]:
                    result = query["result"]
                    
                    # calculate softmax for aggregated answer set
                    scores = np.array([float(answer["score"]) for answer in result["answers"]])
                    #print(scores)
                    softmax = np.exp(scores)/sum(np.exp(scores))
                    for i in range(len(result["answers"])):
                        # convert score (float) into string
                        #result["answers"][i]["score"] = str(result["answers"][i]["score"])
                        # add probability
                        result["answers"][i]["probability"] = str(softmax[i])
                    
                    for tokenized_sample in result["tokenizedSamples"]:
                        # calculate softmax for each tokenized sample
                        scores = np.array([float(answer["score"]) for answer in tokenized_sample["answers"]])
                        #print(scores)
                        softmax = np.exp(scores)/sum(np.exp(scores))
                        sum_softmax = 0.0
                        for i in range(len(tokenized_sample["answers"])):
                            # convert score (float) into string
                            #tokenized_sample["answers"][i]["score"] = str(tokenized_sample["answers"][i]["score"])
                            # add probability
                            tokenized_sample["answers"][i]["probability"] = str(softmax[i])
        return input_dict
                    
    
    def json_to_batch(self, input_dict, no_answer_strategy: str):
        
        batch = {
            "qa_sample_id":[],
            "qa_sample_title":[],
            "qa_sample_query":[],
            "qa_sample_paragraph_id":[],
            "qa_sample_paragraph_title":[],
            "qa_sample_paragraph":[],
            "verbose_output":[]
        }
        if "schemas" not in input_dict:
            raise InvalidRequestException("The request does not contain a list of schemas, i.e., '$.schemas[*]'")
        if not len(input_dict["schemas"]):
            raise InvalidRequestException("The list of schemas, i.e., '$.schemas[*]', must contain at least one schema item")

        for i,schema in enumerate(input_dict["schemas"]):
            if "value" not in schema:
                raise InvalidRequestException("The schema '$.schemas["+str(i)+"]' has no property 'value'")
            if not schema["value"]:
                raise InvalidRequestException("'$.schemas["+str(i)+"].value' must not be empty")
            if "schemaId" not in schema:
                schema["schemaId"] = str(uuid.uuid4())
            if "name" not in schema:
                schema["name"] = "schema "+schema["schemaId"]

            if "queries" not in schema:
                raise InvalidRequestException("The schema '$.schemas["+str(i)+"]' has no list of queries, i.e., '$.schemas["+str(i)+"].queries[*]'")
            if not len(schema["queries"]):
                raise InvalidRequestException("'$.schemas["+str(i)+"].queries[*]' must contain at least one query item")
            
            for j,query in enumerate(schema["queries"]):
                if "value" not in query:
                    raise InvalidRequestException("The query '$.schemas["+str(i)+"].queries["+str(j)+"]' has no property 'value'")
                if not query["value"]:
                    raise InvalidRequestException("'$.schemas["+str(i)+"].queries["+str(j)+"].value' must not be empty")
                if "queryId" not in query:
                    query["queryId"] = str(uuid.uuid4())
                if "name" not in query:
                    query["name"] = "query "+query["queryId"]
                if "verboseOutput" not in query:
                    query["verboseOutput"] = False

                if not self.cache or (self.cache and not self.cache.has(schema["value"],query["value"],no_answer_strategy,query["verboseOutput"])):
                    batch["qa_sample_id"].append(query["queryId"])
                    batch["qa_sample_title"].append(query["name"])
                    batch["qa_sample_query"].append(query["value"])
                    batch["qa_sample_paragraph_id"].append(schema["schemaId"])
                    batch["qa_sample_paragraph_title"].append(schema["name"])
                    batch["qa_sample_paragraph"].append(schema["value"])
                    batch["verbose_output"].append(query["verboseOutput"])
        return batch
        '''
        else:
            return {
                "qa_sample_id":[query["queryId"] for schema in input_dict["schemas"] for query in schema["queries"]],
                "qa_sample_title":[query["name"] for schema in input_dict["schemas"] for query in schema["queries"]],
                "qa_sample_query": [query["value"] for schema in input_dict["schemas"] for query in schema["queries"]],
                "qa_sample_paragraph_id": [schema["schemaId"] for schema in input_dict["schemas"] for _ in schema["queries"]],
                "qa_sample_paragraph_title": [schema["name"] for schema in input_dict["schemas"] for _ in schema["queries"]],
                "qa_sample_paragraph": [schema["value"] for schema in input_dict["schemas"] for _ in schema["queries"]],
                "verbose_output": [query["verboseOutput"] for schema in input_dict["schemas"] for query in schema["queries"]]
            }
        '''
    
    def merge_results_w_input_json(self, input_dict, results, no_answer_strategy:str):
        to_be_cached = []
        for schema in input_dict["schemas"]:
            for query in schema["queries"]:
                if self.cache and self.cache.has(schema["value"],query["value"],no_answer_strategy,query["verboseOutput"]):
                    query["result"] = self.cache.load(schema["value"],query["value"],no_answer_strategy)
                    query["result"]["isCached"]= True
                else:
                    for i in range(len(results["qa_sample_id"])):
                        if results["qa_sample_paragraph_id"][i] == schema["schemaId"] and results["qa_sample_id"][i] == query["queryId"]:
                            result = {
                                "answers": results["answers"][i],
                                "tokenizedSamples": results["tokenized_samples"][i]
                            }
                            query["result"] = result
                            query["result"]["isCached"]= False

                            if self.cache:
                                #BUG-FIX: We MUST NOT store new items in cache until all schemas/queries have been processed.
                                #If we store a new item in cache, an old item might be evicted although it is assumed to be in cache
                                #self.cache.store(schema["value"],query["value"],result,query["verboseOutput"])
                                to_be_cached.append({
                                    "schema": schema["value"],
                                    "query": query["value"],
                                    "result":result,
                                    "verbose":query["verboseOutput"]
                                })
                if "result" not in query:
                    print("Warning!")
        return input_dict, to_be_cached
    
    def store_items_in_cache(self, to_be_cached: dict, no_answer_strategy:str):
        if self.cache:
            for item in to_be_cached:
                self.cache.store(item["schema"],item["query"],no_answer_strategy,item["result"],item["verbose"])
    
    def results_to_json(self, results_dict):
        results = {"results":[]}
        for i in range(len(results_dict["qa_sample_id"])):
            result = {
                "answers": results_dict["answers"][i],
                "tokenizedSamples": results_dict["tokenized_samples"][i]
            }
            results["results"].append(result)
        return results   