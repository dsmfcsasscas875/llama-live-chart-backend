import uvicorn
import os



def main():
    port = int(os.getenv("PORT", 8000))
    # In production (Railway), we must bind to 0.0.0.0
    host = "0.0.0.0" if os.getenv("RAILWAY_ENVIRONMENT") else "127.0.0.1"
    uvicorn.run("app.main:app", host=host, port=port, reload=False if os.getenv("RAILWAY_ENVIRONMENT") else True)

if __name__ == "__main__":
    main()
