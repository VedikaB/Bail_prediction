#<link href="https://bootswatch.com/4/materia/bootstrap.min.css" rel="stylesheet" type="text/css">
from flask import Flask,render_template, flash, redirect , url_for , session ,request, logging
from flask_mysqldb import MySQL
from wtforms import Form, StringField , TextAreaField ,PasswordField , validators
from passlib.hash import sha256_crypt
from functools import wraps
from sklearn.externals import joblib
import flask
import numpy as np 
from flask import Flask, Response, render_template, request
import pandas as pd


app = Flask(__name__, static_url_path='/static')
app.debug = True

svmod1 = joblib.load('main11.pkl')
svmod2 = joblib.load('main12.pkl')
svmod3 = joblib.load('main13.pkl')

#Config MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Svmpvm@12'
app.config['MYSQL_DB'] = 'myflaskapp'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
#init MYSQL
mysql = MySQL(app)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')


class RegisterForm(Form):
    name = StringField('Name',[validators.Length(min=1,max=50)])
    username = StringField('Username',[validators.Length(min=4,max=25)])
    email = StringField('Email',[validators.Length(min=4,max=25)])
    password = PasswordField('Password', [ validators.DataRequired (),validators.EqualTo('confirm',message ='passwords do not match')])
    confirm = PasswordField('Confirm password')
    bookname = StringField('Enter your favorite book name: ',[validators.Length(min=1,max=50)])
    status = StringField('Enter the the status of lawyer: ',[validators.Length(min=6,max=50)])

@app.route('/register', methods=['GET','POST'])
def register():
    form = RegisterForm(request.form)
    if request.method == 'POST' and form.validate():
        name = form.name.data
        email = form.email.data
        username = form.username.data
        password = sha256_crypt.encrypt(str(form.password.data))
        bookname = form.bookname.data
        status = form.status.data

        # Create crusor
        cur = mysql.connection.cursor()

        cur.execute("INSERT INTO users(name,email,username,password,bookname,status) VALUES(%s,%s,%s,%s,%s,%s)",(name,email,username,password,bookname,status))


        # commit to DB
        mysql.connection.commit()
        #close connection
        cur.close()

        flash("You are now Registered and you can login" , 'success')

        redirect(url_for('login'))
    return render_template('register.html',form=form)

#@app.route('/login1as',methods =['GET','POST'])
#def login1as():
#	return render_template('login1as.html')
# user login
@app.route('/login',methods =['GET','POST'])
def login():
    if request.method == 'POST':
        #Get Form Fields
        username = request.form['username']
        password_candidate = request.form['password']
        stat1 = request.form['status']

        # Create cursor

        cur = mysql.connection.cursor()

        #Get user by username

        result = cur.execute("SELECT * FROM users WHERE username = %s" ,[username])
        #stat = cur.execute("SELECT status FROM users WHERE username = %s ",[username])
        if result > 0:
        # Get Stored hash
            data = cur.fetchone()
            password = data['password']
            stat = data['status']

            # Compare Passwords
            if ((sha256_crypt.verify(password_candidate,password)) and (stat == stat1)):
                #Passed
                session['logged_in'] = True
                session['username'] = username

               # flash('You are now logged in ','success')
               # return redirect(url_for('dashboard'))
            if (stat == 'senior'):
                	return render_template('home.html')
            elif (stat == 'junior'):
                	return render_template('dashboard.html')

            else:
                error = 'Wrong username or password'
                return render_template('login.html',error=error)
                #close connection
            cur.close()

        else:
            error = 'Wrong username or password'
            return render_template('login.html',error=error)

    return render_template('login.html')

#check if user logged in

def is_logged_in(f):
    @wraps(f)
    def wrap(*args,**kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized, please login','danger')
            return redirect(url_for('login'))
    return wrap



#logout
@app.route('/logout')
@is_logged_in
def logout():
    session.clear()
   # flash('you are now logged out ','success')
    return render_template('home.html')
# Dashboard
@app.route('/dashboard')
@is_logged_in

def dashboard():
    return render_template('dashboard.html')

@app.route("/predict", methods=['POST'])
def predict():
	if request.method == 'POST':		
#building final list , starting with gender
		gender = request.form['gender']
		if(gender=='male'):
			gen1=1
		else:
			gen1=0
		injuiry = request.form['injury']
		if(injuiry=='No'):
			in1=0
		else:
			in1=1
		coa = request.form['CoAccused']
		if(coa=='Yes'):
			co1=1
		else:
			co1=0
		sob = request.form['SOB']
		if(sob=='Yes'):
			sob1=1
		else:
			sob1=0
		intention = request.form['intention']
		if(intention=='Yes'):
			i1=1
		else:
			i1=0
		ew = request.form['Eyewitness']
		if(ew=='Yes'):
			ew1=1
		else:
			ew1=0
		his = request.form['history']
		if(his=='Yes'):
			his1=1
		else:
			his1=0
		lop = request.form['loopwhole']
		if(lop=='Yes'):
			lop1=1
		else:
			lop1=0
		noa = request.form['NOA']
		if(noa=='Very Serious'):
			ns=0
			s=0
			vs=1
		if(noa=='serious'):
			ns=0
			s=1
			vs=0
		else:
			ns=1
			s=0
			vs=0
		poc = request.form['period']
		diff = request.form['differences']
		if(diff=='Yes'):
			diff1=1
		else:
			diff1=0
		cond = request.form['condition']
		if(cond=='Stable'):
			cond1=0
		else:
			cond1=1
		loop = request.form['loopwhole']
		if(loop=='Yes'):
			loop1=1
		else:
			loop1=0
		acc = request.form['Present']
		if(acc=='Yes'):
			acc1=1
		else:
			acc1=0
		proof = request.form['Proof']
		if(proof=='Yes'):
			proof1=1
		else:
			proof1=0
		chargesheet = request.form['chargesheet']
		if(chargesheet=='Yes'):
			chargesheet1=1
		else:
			chargesheet1=0
		
		#return render_template('result1.html')
		
@app.route("/result1", methods=['POST'])
def result1():
    if request.method == 'POST':		
#building final list , starting with gender
		gender = request.form['gender']
		if(gender=='male'):
			gen1=1
		else:
			gen1=0
		injuiry = request.form['injury']
		if(injuiry=='No'):
			in1=0
		else:
			in1=1
		coa = request.form['CoAccused']
		if(coa=='Yes'):
			co1=1
		else:
			co1=0
		sob = request.form['SOB']
		if(sob=='Yes'):
			sob1=1
		else:
			sob1=0
		intention = request.form['intention']
		if(intention=='Yes'):
			i1=1
		else:
			i1=0
		ew = request.form['Eyewitness']
		if(ew=='Yes'):
			ew1=1
		else:
			ew1=0
		his = request.form['history']
		if(his=='Yes'):
			his1=1
		else:
			his1=0
		lop = request.form['loopwhole']
		if(lop=='Yes'):
			lop1=1
		else:
			lop1=0
		noa = request.form['NOA']
		if(noa=='Very Serious'):
			ns=0
			s=0
			vs=1
		if(noa=='serious'):
			ns=0
			s=1
			vs=0
		else:
			ns=1
			s=0
			vs=0
		poc = request.form['period']
		diff = request.form['differences']
		if(diff=='Yes'):
			diff1=1
		else:
			diff1=0
		cond = request.form['condition']
		if(cond=='Stable'):
			cond1=0
		else:
			cond1=1
		loop = request.form['loopwhole']
		if(loop=='Yes'):
			loop1=1
		else:
			loop1=0
		acc = request.form['Present']
		if(acc=='Yes'):
			acc1=1
		else:
			acc1=0
		proof = request.form['Proof']
		if(proof=='Yes'):
			proof1=1
		else:
			proof1=0
		chargesheet = request.form['chargesheet']
		if(chargesheet=='Yes'):
			chargesheet1=1
		else:
			chargesheet1=0

   #data=np.array(list([gen1,in1,co1,sob1,i1,ew1,his1,lop1,ns,s,vs,poc,diff1,cond1,loop1,acc1,proof1,chargesheet1]),np.float)
   #print(data)
   #data=data.reshape(1, -1)
   data1=np.array(list([in1,sob1,i1,ew1,his1,lop1,ns,s,vs,cond1,loop1,proof1]),np.float)
   data1=data.reshape(1, -1)

	prediction = svmod2.predict(data1)
	per_pred=svmod2.predict_proba(data1)
	print('Prediction : ')
	print(prediction)
	if prediction == 0:
			prediction = 'The Bail will be Granted!'
	else:
			prediction = 'The Bail will not be Granted! :)'

	print('percentage:')
	print(per_pred)

	return flask.render_template('result1.html', label = prediction , label1 = per_pred)
 

 


	



if __name__ =='__main__':
    app.secret_key='secret123'
    app.run()
