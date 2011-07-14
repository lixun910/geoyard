import matplotlib

def handle(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/plain')])
    return [environ['PATH_TRANSLATED']]