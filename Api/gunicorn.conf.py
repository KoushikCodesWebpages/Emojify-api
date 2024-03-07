bind = "0.0.0.0:5000" #(ports)
workers = 2  

wsgi_app = "app:app"
#command
#waitress-serve --listen=*:5000 app:app
#C:\Users\Koush\OneDrive\Desktop\virtusa_project\Api