# Copyright 2016-2018 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/kiosk-data-processing/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""REST API for data pre-processing and post-processing transformations"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from decouple import config
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np

from data_processing.utils import get_function


app = Flask(__name__)

CORS(app)


if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)


@app.before_first_request
def setup_logging():
    """Set up logging to send INFO to stderr"""
    if not app.debug:
        # In production mode, add log handler to sys.stderr.
        app.logger.setLevel(logging.INFO)


@app.route('/health', methods=['GET'])
def health():
    """Health check end-point"""
    return jsonify({'status': 'OK'}), 200


@app.route('/process/<process_type>/<function_name>', methods=['POST'])
def process(process_type, function_name):
    """Process data according to the process type and function name.
    e.g. `/process/pre/normalize` or `/process/post/watershed`
    # Arguments:
        process_type: pre or post processing
        function_name: name of function to apply.
    # Returns:
        transformed data as JSON
    """
    try:
        # first, find the requested processing function
        F = get_function(process_type, function_name)
    except KeyError as err:
        return jsonify({'error': '{} Not Found'.format(err)}), 404

    try:
        # second, verify the post request data
        request_json = request.get_json(force=True)
    except Exception as err:  # pylint: disable=broad-except
        errmsg = 'Malformed JSON: {}'.format(err)
        app.logger.error(errmsg)
        return jsonify({'error': errmsg}), 400

    try:
        images = [np.array(i['image']) for i in request_json['instances']]
    except Exception as err:  # pylint: disable=broad-except
        errmsg = 'Failed to convert JSON response to np.array due to %s: %s'
        app.logger.error(errmsg % (type(err).__name__), err)
        return jsonify({'error': errmsg}), 400

    try:
        processed = []
        for image in images:
            app.logger.debug('%s %s-processing image with shape %s',
                             function_name.capitalize(), process_type,
                             image.shape)

            processed_img = F(image)
            processed.append(processed_img.tolist())

            app.logger.debug('%s %s-processed image with shape %s',
                             function_name.capitalize(), process_type,
                             processed_img.shape)

        return jsonify({'processed': processed}), 200
    except Exception as err:  # pylint: disable=broad-except
        errmsg = '{} applying {} {}-processing: {}'.format(
            type(err).__name__, function_name, process_type, err)
        app.logger.error(errmsg)
        return jsonify({'error': errmsg}), 500


if __name__ == '__main__':
    DEBUG = config('DEBUG', default=False, cast=bool)
    LISTEN_PORT = config('LISTEN_PORT', default=8080, cast=int)

    app.run(debug=DEBUG, host='0.0.0.0', port=LISTEN_PORT)
