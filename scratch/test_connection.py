import sys
sys.path.insert(0, "/Users/sandaruariyaratne/Downloads/algo_trader/.venv/lib/python3.10/site-packages")

import asyncio
import ccxt.pro as ccxtpro

async def test_exchange(exchange_id):
    print(f"Testing connectivity to {exchange_id} ...")
    exchange = getattr(ccxtpro, exchange_id)()
    try:
        markets = await exchange.load_markets()
        print(f"  [SUCCESS] {exchange_id} is reachable! Loaded {len(markets)} markets.")
        await exchange.close()
        return True
    except Exception as exc:
        print(f"  [FAILED] {exchange_id} failed: {exc}")
        try:
            await exchange.close()
        except:
            pass
        return False

async def main():
    exchanges = ["binance", "bybit", "okx", "kraken", "gateio"]
    for ex in exchanges:
        await test_exchange(ex)

if __name__ == "__main__":
    asyncio.run(main())
