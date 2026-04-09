import asyncio
import httpx
import json

async def test_backend():
    async with httpx.AsyncClient() as client:
        try:
            # Test Root
            print("Testing Root...")
            resp = await client.get("http://127.0.0.1:8000/")
            print(f"Root Status: {resp.status_code}")
            print(f"Root Response: {resp.text}")

            # Test Stock Sentiment (BTC as example)
            print("\nTesting Stock Sentiment (BTC)...")
            # Set a longer timeout because of LLM processing
            resp = await client.get("http://127.0.0.1:8000/api/v1/stock/sentiment/BTC", timeout=30.0)
            print(f"Sentiment Status: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                print(f"Overall Sentiment: {data.get('overall_sentiment')}")
                print(f"Sentiment Score: {data.get('sentiment_score')}")
                print(f"Number of headlines analyzed: {len(data.get('news', []))}")
            else:
                print(f"Sentiment Error: {resp.text}")

        except Exception as e:
            print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_backend())
