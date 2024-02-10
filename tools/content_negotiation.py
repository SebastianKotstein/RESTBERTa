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

from flask import request, Response
from werkzeug.exceptions import UnsupportedMediaType, NotAcceptable
from functools import wraps

def consumes(*mime_types):
    def decorated(fn):
        @wraps(fn) # preserves name of decorated function (required for routing in flask)
        def inner(*args, **kwargs):
            if request.mimetype not in mime_types:
                raise UnsupportedMediaType()
            #kwargs["content_type"] = request.mimetype
            return fn(*args, **kwargs)
        return inner
    return decorated

def produces(*mime_types, default_mime_type = None, pass_negotiated_mime_type = False, allow_empty_accept_header = True):
    def decorated(fn):
        @wraps(fn) # preserves name of decorated function (required for routing in flask)
        def inner(*args, **kwargs):
            accepted = set(request.accept_mimetypes.values())
            if allow_empty_accept_header and not accepted:
                res = fn(*args, **kwargs)
                if default_mime_type:
                    res.headers["Content-Type"] = default_mime_type
                return res
            else:
                supported = set(mime_types)
                if len(accepted & supported) == 0:
                    raise NotAcceptable()
                if pass_negotiated_mime_type:
                    for accepted_mime_type in accepted:
                        if accepted_mime_type in supported:
                            kwargs["accept"] = accepted_mime_type
                            break
                res = fn(*args, **kwargs)
                if default_mime_type:
                    res.headers["Content-Type"] = default_mime_type
                return res
        return inner
    return decorated