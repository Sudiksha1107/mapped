docker build -t main .
docker run --rm -p 8000:8000 --env-file .env main
