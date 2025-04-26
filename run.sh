#!/bin/bash
nohup python -u -m streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false --server.port 8090 > ~/LangGraphChatBot.log 2>&1 &
