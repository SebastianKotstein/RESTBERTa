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


class LRUCache:

    def __init__(self, max_size = 1000, debug = False) -> None:
        self.max_size = max_size
        self.access_counter = 0

        self.results = dict()
        self.verbose_info = dict()
        self.access_counters = dict()
        self.debug = debug

    def has(self, schema: str, query: str, verbose: bool):
        key = self.generate_key(schema,query)
        if verbose:
            if key in self.results and self.verbose_info[key]:
                return True
            else:
                return False
        else:
            if key in self.results:
                return True
            else:
                return False

    def load(self, schema: str, query: str):
        key = self.generate_key(schema,query)
        if self.has(schema,query,False):
            if self.debug:
                print("Load "+key)
            result = self.results[key]
            self.access_counter+=1
            self.access_counters[key] = self.access_counters
            return result.copy()
        else:
            if self.debug:
                print("Load "+key+" - not found")
            None

    def store(self, schema: str, query: str, result, verbose: bool):
        if not self.has(schema,query,False) and len(self.results.values()) == self.max_size:
            sorted_keys = sorted(self.access_counters)
            self.evict(sorted_keys[0])
        
        key = self.generate_key(schema,query)
        if self.debug:
            print("Store "+key)    
        self.results[key] = result.copy()
        self.verbose_info[key] = verbose
        self.access_counter+=1
        self.access_counters[key] = self.access_counter


    def evict(self, schema: str, query: str):
        if self.has(schema,query,False):
            key = self.generate_key(schema,query)
            self.evict(key)
            
    def evict(self, key: str):
        if self.debug:
            print("Evict "+key)
        del self.results[key]
        del self.verbose_info[key]
        del self.access_counters[key]

    def evict_all(self):
        if self.debug:
            print("Evict all")
        self.access_counter = 0
        self.results.clear()
        self.verbose_info.clear()
        self.access_counters.clear()

    def generate_key(self, schema: str, query: str):
        return "{'schema':'"+schema+"', 'query':'"+query+"'}"

    
    

