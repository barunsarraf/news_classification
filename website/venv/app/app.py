from flask import Flask, render_template, request

import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


app = Flask(__name__)

@app.route('/')

def index():
	

	return render_template("index.html")



@app.route('/send', methods=['GET','POST'])

def send():
	if request.method == 'POST':
		art = request.form['article_value']

		category_list_dict_rev = {0:'BUSINESS', 1:'TECHNOLOGY', 2:'ENTERTAINMENT', 3:'HEALTH'}
		#category_list = ["sport", "world", "us", "business", "health", "entertainment", "sci_tech"]
		docs_new = art
		docs_new = [docs_new]

		loaded_vec = CountVectorizer(vocabulary=pickle.load(open("count_vec.pkl", "rb")))
		loaded_tfidf = pickle.load(open("tfidf_new.pkl","rb"))
		loaded_model = pickle.load(open("svm_new.pkl","rb"))

		X_new_counts = loaded_vec.transform(docs_new)
		X_new_tfidf = loaded_tfidf.transform(X_new_counts)
		predicted = loaded_model.predict(X_new_tfidf)


		answer=category_list_dict_rev[predicted[0]]


		return render_template('index.html',result=answer)


	return render_template('index.html')	



if __name__ == '__main__':
	app.run()