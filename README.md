Start the project by using the following command

pip install -r requirements.txt

Make sure to add your open ai key before running the code in line number 35 of main.py

python main.py

Once the code has been started start postman and create a post api with the following url

http://127.0.0.1:5000/recommend

Add the json in response as

{
  "query": "test query"
}
