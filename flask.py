from flask import Flask, request #import main Flask class and request object

app = Flask(__name__) #create the Flask app


@app.route('/recommend')
def recommend():
    unikey = request.args.get('unikey') #if key doesn't exist, returns None

    return '''<h1>The unikey is: {}</h1>'''.format(unikey)
