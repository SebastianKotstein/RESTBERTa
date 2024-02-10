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

import tensorflow as tf
from transformers import TFAutoModelForQuestionAnswering

class QAModel:
    def __init__(self, checkpoint, batch_size = None) -> None:
        print(tf.config.list_physical_devices('GPU'))
        self.model = TFAutoModelForQuestionAnswering.from_pretrained(checkpoint)
        self.batch_size = batch_size
        
    
            
    def predict(self, batched_samples):
        """
        Converts the passed batch of tokenized samples into a tensor that is feed into the passed transformer model for prediction.
        The method returns the model's output as well as the number of input samples.
        
        Parameters
        ----------
        batched_samples : [dict()]
            Batch of samples where each sample is a dictionary have the fields 'attention_mask' and 'input_ids'
            
        Returns
        -------
        The output of the model (first return parameter) and the number of input samples (second return parameter)
        """
        
        batch_counter = 0
        for i in range(len(batched_samples["input_ids"])):
            # create input tensor for each sample of batch:
            # attention mask is a binary tensor so that the model knows to which token it has to attend to (typically 0 for padded indices)
            attention_mask_t = tf.constant([batched_samples["attention_mask"][i]])
            input_ids_t = tf.constant([batched_samples["input_ids"][i]])
            
            if batch_counter == 0:
                batch = dict()
                batch["attention_mask"] = attention_mask_t
                batch["input_ids"] = input_ids_t
            else:
                batch["attention_mask"] = tf.concat([batch["attention_mask"], attention_mask_t],axis=0)
                batch["input_ids"] = tf.concat([batch["input_ids"],input_ids_t],axis=0)
            batch_counter+=1
        
        output = self.model.predict(batch, batch_size = self.batch_size, verbose=0)
        return output, batch_counter


    

    