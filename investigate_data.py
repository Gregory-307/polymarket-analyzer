"""Investigate raw market data - no assumptions, just facts."""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import Credentials
from src.adapters.polymarket import PolymarketAdapter


async def investigate():
    """Raw data investigation."""

    creds = Credentials.from_env()
    adapter = PolymarketAdapter(credentials=creds)
    await adapter.connect()
    markets = await adapter.get_markets(active_only=True, limit=200)
    await adapter.disconnect()

    print("=" * 70)
    print("RAW DATA INVESTIGATION")
    print("=" * 70)

    # Collect high-probability markets
    high_prob_markets = []
    for m in markets:
        yes_p = m.yes_price
        no_p = m.no_price
        high_p = max(yes_p, no_p)

        if high_p >= 0.90:
            side = "YES" if yes_p > no_p else "NO"
            high_prob_markets.append({
                "question": m.question[:60],
                "yes_price": yes_p,
                "no_price": no_p,
                "high_prob": high_p,
                "side": side,
                "volume": m.volume or 0,
                "liquidity": m.liquidity or 0,
            })

    print(f"\nTotal markets: {len(markets)}")
    print(f"Markets with >90% probability: {len(high_prob_markets)}")

    # Sort by probability
    high_prob_markets.sort(key=lambda x: x["high_prob"], reverse=True)

    print("\n" + "=" * 70)
    print("TOP 20 HIGH-PROBABILITY MARKETS")
    print("=" * 70)

    for i, m in enumerate(high_prob_markets[:20], 1):
        print(f"{i:2}. {m['question']}")
        print(f"    {m['side']} @ {m['high_prob']:.2%} | Vol: ${m['volume']:,.0f} | Liq: ${m['liquidity']:,.0f}")
        print()

    # Distribution
    print("=" * 70)
    print("PROBABILITY DISTRIBUTION (>90% markets)")
    print("=" * 70)

    buckets = [
        ("99%+", 0.99, 1.01),
        ("98-99%", 0.98, 0.99),
        ("97-98%", 0.97, 0.98),
        ("96-97%", 0.96, 0.97),
        ("95-96%", 0.95, 0.96),
        ("94-95%", 0.94, 0.95),
        ("93-94%", 0.93, 0.94),
        ("92-93%", 0.92, 0.93),
        ("91-92%", 0.91, 0.92),
        ("90-91%", 0.90, 0.91),
    ]

    for label, low, high in buckets:
        count = len([m for m in high_prob_markets if low <= m["high_prob"] < high])
        bar = "#" * count
        print(f"{label:>10}: {count:>3} {bar}")

    # THE CRITICAL QUESTION: What is my edge calculation actually based on?
    print("\n" + "=" * 70)
    print("HONEST ASSESSMENT OF EDGE METHODOLOGY")
    print("=" * 70)

    print("""
MY EDGE CLAIM IS BASED ON:

1. RESEARCH ASSUMPTION (NOT VALIDATED WITH ACTUAL DATA):
   - Papers claim high-prob outcomes resolve MORE than priced
   - E.g., 95% priced markets resolve YES ~97-98% of the time
   - This would give 2-3% edge

2. THE PROBLEM:
   - I do NOT have historical resolution data from Polymarket
   - I cannot actually verify if this bias exists on Polymarket
   - My "edge" calculation is: fair_value = price + 0.03 (arbitrary!)
   - This is an ASSUMPTION, not a measured edge

3. WHAT WE ACTUALLY KNOW:
   - Current market prices (YES)
   - Volume and liquidity (YES)
   - Historical resolution rates (NO - we don't have this data)
   - Actual edge (NO - this is a guess)

4. TO PROPERLY VALIDATE THIS STRATEGY:
   - Need historical data of resolved markets
   - Compare market prices at time of betting vs actual outcomes
   - Calculate actual edge = resolution_rate - market_price
   - We have NOT done this

HONEST CONCLUSION:
The favorite-longshot bias is well-documented in academic literature,
but I have NOT validated it exists on Polymarket with actual data.
The 2-3% edge is an ASSUMPTION based on research, not measurement.
""")

    # What CAN we say for certain?
    print("=" * 70)
    print("WHAT WE CAN SAY FOR CERTAIN")
    print("=" * 70)

    total_vol = sum(m["volume"] for m in high_prob_markets)
    total_liq = sum(m["liquidity"] for m in high_prob_markets)

    print(f"""
FACTS (verified from live data):
- {len(high_prob_markets)} markets priced at >90% probability
- ${total_vol:,.0f} total volume in these markets
- ${total_liq:,.0f} total liquidity available
- Prices shown are current market prices

ASSUMPTIONS (from research, not verified):
- High-probability markets may be underpriced by 2-5%
- Win rate should be higher than market price implies
- This is based on Kahneman/Tversky research in OTHER markets

RISK:
- If the bias doesn't exist on Polymarket, there is NO edge
- We'd be paying transaction costs to bet on fairly-priced markets
- Black swan events (the 5%) would result in 100% loss of position
""")

    return high_prob_markets


if __name__ == "__main__":
    asyncio.run(investigate())
