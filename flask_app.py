# from flask import Flask

# app = Flask(__name__)

# @app.route('/')

# def hello_world():
#     return 'Hello from Naeem!'

#---------------

from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

# import sys

app = Flask(__name__)

@app.route('/tryhome')
def tryhome():
	return render_template('home_2.html')

@app.route('/sentiment')
def sentiment():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("/home/naeemhussien/mysite/youtube_sentiment/YoutubeSpamMergedData.csv")
	df_data = df[["CONTENT","CLASS"]]
	# Features and Labels
	df_x = df_data['CONTENT']
# 	df_y = df_data.CLASS
    # Extract Feature With CountVectorizer
	corpus = df_x
	cv = CountVectorizer()
	X = cv.fit_transform(corpus) # Fit the Data

# 	X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
# 	Naive Bayes Classifier
# 	clf = MultinomialNB()
# 	clf.fit(X_train,y_train)
# 	clf.score(X_test,y_test)
	#Alternative Usage of Saved Model
	ytb_model = open("/home/naeemhussien/mysite/youtube_sentiment/clf_model.pkl","rb")
	clf = joblib.load(ytb_model)
# 	ytb1_model = open("/home/naeemhussien/mysite/youtube_sentiment/cv_model.pkl","rb")
    # cv = joblib.load(ytb1_model)

	if request.method == 'POST':
		comment = request.form['comment']
# 		data = [comment]
		vect = cv.transform([comment]).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)

##------------

# from flask import Flask,render_template,url_for,request
# import pandas as pd
# import pickle
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.externals import joblib
# from sklearn.model_selection import train_test_split

# # import sys

# app = Flask(__name__)

# @app.route('/')
# def home():
# 	return render_template('home.html')

# @app.route('/predict',methods=['POST'])
# def predict():
# 	df= pd.read_csv("/home/naeemhussien/mysite/youtube_sentiment/YoutubeSpamMergedData.csv")
# 	df_data = df[["CONTENT","CLASS"]]
# # 	Features and Labels
# 	df_x = df_data['CONTENT']
# # 	df_y = df_data.CLASS
#     # Extract Feature With CountVectorizer
# 	corpus = df_x
# 	cv = CountVectorizer()
# 	X = cv.fit_transform(corpus) # Fit the Data
# 	cv_model = open("/home/naeemhussien/mysite/youtube_sentiment/cv_model.pkl","rb")
#     # cv_final = pickle.load(cv_model)

# # 	X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
# # 	Naive Bayes Classifier
# # 	clf = MultinomialNB()
# # 	clf.fit(X_train,y_train)
# # 	clf.score(X_test,y_test)
# 	#Alternative Usage of Saved Model
# 	ytb_model = open("/home/naeemhussien/mysite/youtube_sentiment/clf_model.pkl","rb")
# 	clf = joblib.load(ytb_model)



# 	if request.method == 'POST':
# 		comment = request.form['comment']
# # 		data = [comment]
# 		vect = cv_final.transform([comment]).toarray()
# 		my_prediction = clf.predict(vect)
# 	return render_template('result.html',prediction = my_prediction)



# if __name__ == '__main__':
# 	app.run(debug=True)

