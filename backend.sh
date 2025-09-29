docker build -t bot .
docker run --rm -it --env-file .env bot
streamlit run bot.py