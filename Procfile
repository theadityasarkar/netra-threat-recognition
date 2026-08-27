# NETRA — process definition for Heroku / Railway / Dokku-style PaaS.
#
# $PORT is injected by the platform and MUST be honoured, or the health check
# never passes and the dyno is killed. --server.address=0.0.0.0 makes the app
# reachable from outside the container; --server.headless suppresses the
# "open a browser" prompt that would otherwise block start-up.
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false
