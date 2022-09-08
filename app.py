from flask import Flask,render_template,request
import pickle
import numpy as np
import project 
ap=pickle.load(open('model.pkl','rb'))
print(project.x)
app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("index.html")
    
@app.route('/pr',methods=["POST"])
def predict():
    
    if request.method == "POST":
        retailer_id = int(request.form["retailer"])  
        print(retailer_id)
        p_date = int(request.form["date"])
        print(p_date)
        n_orders = int(request.form["orders"])
        print(n_orders)
        t_rev = int(request.form["rev"])
        print(t_rev)
        rec_c = int(np.where(p_date <= project.recency_table['recency'].quantile(0.25),1,np.where(p_date<=project.recency_table['recency'].quantile(0.50),2,np.where(p_date<=project.recency_table['recency'].quantile(0.75),3,4))))
        fre_c = int(np.where(n_orders<=project.frequence['number_of_transaction'].quantile(0.25),1,np.where(n_orders<=project.frequence['number_of_transaction'].quantile(0.50),2,np.where(n_orders<=project.frequence['number_of_transaction'].quantile(0.75),3,4))))
        rev_c = int(np.where( t_rev<=project.revenue['value'].quantile(0.25),1,np.where( t_rev<=project.revenue['value'].quantile(0.50),2,np.where( t_rev<=project.revenue['value'].quantile(0.75),3,4))))
        sc= rec_c + fre_c + rev_c
        #p=project.pred([retailer_id,p_date,rec_c,n_orders,fre_c,t_rev,rev_c,sc])
        pred= ap.predict(np.array([[retailer_id,p_date,rec_c,n_orders,fre_c,t_rev,rev_c,sc]]))
        
    
    return render_template("index.html",out=pred)


if __name__ == '__main__':
   app.run()
   
   
   
   
   
   

