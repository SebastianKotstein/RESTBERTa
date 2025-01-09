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


from flask import Flask, request, jsonify, render_template, Response, url_for, send_from_directory
from pipeline.pipeline import Pipeline, InvalidRequestException
from pipeline.lru_cache import LRUCache
import json
from datetime import datetime
from werkzeug.exceptions import HTTPException, BadRequest, NotFound
from flask_swagger_ui import get_swaggerui_blueprint
from representations import *
from content_negotiation import *

import os

app = Flask(__name__,static_folder='static')

MODEL_PM = "SebastianKotstein/restberta-qa-parameter-matching"
MODEL_ED = "SebastianKotstein/restberta-qa-endpoint-discovery"
MODEL_ED_PM = "SebastianKotstein/restberta-qa-pm-ed"

#model = os.getenv("MODEL",default=MODEL_PM)
if "MODEL" in os.environ:
    model = os.environ["MODEL"]
else:
    model = MODEL_PM

#best_size = os.getenv("BEST_SIZE",default=20)
if "BEST_SIZE" in os.environ:
    best_size = int(os.environ["BEST_SIZE"])
else:
    best_size = 20

#cache_size = os.getenv("CACHE",default=100)   
if "CACHE" in os.environ:
    cache_size = int(os.environ["CACHE"])
else:
    cache_size = 100

if "TOKEN" in os.environ:
    token = os.environ["TOKEN"]
else:
    token = None

print("Model: ",model)
if cache_size:
    cache = LRUCache(cache_size,False)
else:
    cache = None
pipeline = Pipeline(model,best_size,cache,token)


SWAGGER_URL = '/docs' 
OPEN_API_FILE = '/openapi.yml'  

# Call factory function to create our blueprint
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
    OPEN_API_FILE,
    config={  # Swagger UI config overrides
        'app_name': "Test application"
    },
    # oauth_config={  # OAuth config. See https://github.com/swagger-api/swagger-ui#oauth2-configuration .
    #    'clientId': "your-client-id",
    #    'clientSecret': "your-client-secret-if-required",
    #    'realm': "your-realms",
    #    'appName': "your-app-name",
    #    'scopeSeparator': " ",
    #    'additionalQueryStringParams': {'test': "hello"}
    # }
)

if model == MODEL_PM:
    header = "RESTBERTa Parameter Matching"
    paragraph = "Schema"
    answer = "Parameter"
    example = "auth.key location.city location.city_id location.country location.lat location.lon location.postal_code state units"
elif model == MODEL_ED:
    header = "RESTBERTa Endpoint Discovery"
    paragraph = "List of Endpoints"
    answer = "Endpoint"
    example = "auth.post users.get users.post users.{userId}.address.get users.{userId}.address.put users.{userId}.delete users.{userId}.get users.{userId}.put"
else:
    header = "RESTBERTa"
    paragraph = "Schema"
    answer = "Answer"
    example = "auth.key location.city location.city_id location.country location.lat location.lon location.postal_code state units"
    
app.register_blueprint(swaggerui_blueprint)
app.config['JSON_SORT_KEYS'] = False

@app.route("/predict",methods=["POST"])
@produces(MIME_TYPE_APPLICATION_JSON,MIME_TYPE_RESULTS_V1_JSON, default_mime_type=MIME_TYPE_RESULTS_V1_JSON)
@consumes(MIME_TYPE_SCHEMAS_V1_JSON,MIME_TYPE_APPLICATION_JSON)
def api():
    args = request.args
    top_answers_n = None
    no_answer_strategy = None

    suppress_duplicates = False
    if "duplicates" in args and args["duplicates"] == "suppress":
        suppress_duplicates = True

    if "top" in args and args["top"]:
        top_answers_n = int(args["top"])

    if "no-answer-strategy" in args:
        no_answer_strategy = args["no-answer-strategy"]
    else:
        no_answer_strategy = "ignore"
    if no_answer_strategy != "treshold" and no_answer_strategy != "ignore":
        raise BadRequest(description = "Invalid value for query parameter 'no-answer-strategy'. Allowed values are 'ignore' and 'treshold'.")

    try:
        response_payload = pipeline.process(request.json,top_answers_n,suppress_duplicates,no_answer_strategy)
        response_payload["_links"] = [
            {
                "rel":"prediction",
                "href": url_for("api")
            },
            {
                "rel":"base",
                "href": url_for("base")
            }
        ]

        response = jsonify(response_payload)
        #response.mimetype=MIME_TYPE_RESULTS_V1_JSON
        return response
    except InvalidRequestException as e:
        raise BadRequest(description = e.message)


@app.route("/",methods=["GET"])
@produces(MIME_TYPE_APPLICATION_XHTML_XML,MIME_TYPE_TEXT_HTML,MIME_TYPE_HYPERMEDIA_V1_JSON,MIME_TYPE_APPLICATION_JSON)
def base():
    if MIME_TYPE_APPLICATION_XHTML_XML in list(request.accept_mimetypes.values()) or MIME_TYPE_TEXT_HTML in list(request.accept_mimetypes.values()):
        return render_template("index.html", header=header, paragraph=paragraph, answer=answer, example=example)
    else:
        response = jsonify({
            "_links":[
                {
                    "rel":"prediction",
                    "href": url_for("api")
                },
                {
                    "rel":"self",
                    "href": url_for("base")
                }
            ]
        })
        response.mimetype = MIME_TYPE_HYPERMEDIA_V1_JSON
        return response
    
@app.route("/cache",methods=["GET"])
@produces(MIME_TYPE_CACHE_SETTINGS_V1_JSON,MIME_TYPE_APPLICATION_JSON)
def get_cache_settings():
    payload = dict()
    if cache:
        payload["isEnabled"] = True
        payload["cacheSize"] = cache_size
    else:
        payload["isEnabled"] = False

    payload["_links"] = []
    if cache:
        payload["_links"].append({
            "rel":"cached-items",
            "href":url_for("get_cached_items")
        })
    payload["_links"].append({
        "rel":"base",
        "href": url_for("base")
    })
    payload["_links"].append({
        "rel":"self",
        "href": url_for("get_cache_settings")
    })
    
    response = jsonify(payload)
    response.mimetype = MIME_TYPE_CACHE_SETTINGS_V1_JSON
    return response

@app.route("/cache/items",methods=["GET"])
@produces(MIME_TYPE_CACHED_ITEMS_V1_JSON,MIME_TYPE_APPLICATION_JSON)
def get_cached_items():
    payload = dict()
    if cache:
        payload = dict()
        payload["cachedItems"] = []
        for key in cache.results.keys():
            id = cache.keys_to_ids[key]
            payload["cachedItems"].append({
                "id":id,
                "key":key,
                "priority":cache.access_counters[key],
                "isVerbose":cache.verbose_info[key],
                "_links":[
                    {
                        "rel":"item",
                        "href":url_for("get_cached_item",id=id)
                    }
                ]
            })
        response = jsonify(payload)
        response.mimetype = MIME_TYPE_CACHED_ITEMS_V1_JSON
        return response
    else:
        raise NotFound("The requested resource does not exist, since caching is disabled.")

@app.route("/cache/items/<id>",methods=["GET"])
@produces(MIME_TYPE_CACHED_ITEMS_V1_JSON, MIME_TYPE_APPLICATION_JSON)
def get_cached_item(id):
    if cache:
        if id in cache.ids_to_keys.keys():
            payload = dict()
            payload["id"] = id
            key = cache.ids_to_keys[id]
            payload["key"] = key
            payload["priority"] = cache.access_counters[key]
            payload["isVerbose"] = cache.verbose_info[key]
            payload["data"] = cache.results[key]
            payload["_links"] = [
                    {
                        "rel":"collection",
                        "href":url_for("get_cached_items")
                    },
                    {
                        "rel":"self",
                        "href":url_for("get_cached_item",id=id)
                    }
                ]
            response = jsonify(payload)
            response.mimetype = MIME_TYPE_CACHED_ITEM_V1_JSON
            return response
        else:
            raise("The requested cache item with ID '"+id+"' does not exist.")
    else:
       raise NotFound("The requested resource does not exist, since caching is disabled.") 
    

@app.route('/openapi.yml')
def send_docs():
    return send_from_directory(app.static_folder, 'OpenAPI.yml')

@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "status": e.code,
        "error": e.name,
        "message": e.description,
        "path": request.url,
        "_links":[
                {
                    "rel":"base",
                    "href": url_for("base")
                },
                {
                    "rel":"docs",
                    "href": SWAGGER_URL
                }
            ]
    })
    response.content_type = MIME_TYPE_ERROR_V1_JSON
    return response

'''
if __name__ == '__main__':
    pipeline = Pipeline("SebastianKotstein/restberta-qa-parameter-matching",20,2)

    input = {
        "schemas":[
            {
                "schemaId": "s1",
                "name": "testSchema",
                "value": "auth.key location.city location.city_id location.country location.lat location.lon location.postal_code state units", 
                "queries":[
                    {
                        "queryId": "q1",
                        "name": "first query",
                        "value": "The ZIP",
                        "verboseOutput":False
                    },
                    {
                        "queryId": "q2",
                        "name": "second query",
                        "value": "The auth token",
                        "verboseOutput":False
                    }
                    
                ]
            },
            {
                "schemaId": "s2",
                "name": "schemaWoQueries",
                "value": "none",
                "queries":[]
            }
        ]
    }

    results = pipeline.process(input)
    print(json.dumps(results, indent=2))
'''
