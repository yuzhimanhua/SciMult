# Copyright (c) 2023 Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import importlib

from dpr.models import BIENCODER_INITIALIZERS, init_comp


def init_hf_any_biencoder(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError("Please install transformers lib")
    from .hf_models_contrib import get_any_biencoder_components

    return get_any_biencoder_components(args, **kwargs)


BIENCODER_INITIALIZERS_V2 = {
    **BIENCODER_INITIALIZERS,
    "hf_any": init_hf_any_biencoder,
}


def init_biencoder_components_v2(encoder_type: str, args, **kwargs):
    return init_comp(BIENCODER_INITIALIZERS_V2, encoder_type, args, **kwargs)
