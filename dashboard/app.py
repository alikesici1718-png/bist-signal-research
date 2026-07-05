from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    clusters = get_clusters()  # You need to implement this function to fetch the clusters
    return render_template('index.html', clusters=clusters)

if __name__ == '__main__':
    app.run(debug=True)
