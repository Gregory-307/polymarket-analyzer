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

    # Get markets in 90%+ zone
    high_prob = []
    for m in markets:
        high_p = max(m.yes_price, m.no_price)
        if high_p >= 0.90:
            side = "YES" if m.yes_price > m.no_price else "NO"
            # Get the token ID for the high-prob side
            tokens = m.raw.get("tokens", [])
            token_id = None
            for t in tokens:
                outcome = t.get("outcome", "").upper()
                if outcome == side:
                    token_id = t.get("token_id")
                    break

            # Debug: if first few markets, print token info
            if len(high_prob) < 3 and 0.90 <= high_p < 0.96:
                print(f"DEBUG: {m.question[:40]}")
                print(f"  tokens: {tokens[:2] if tokens else 'EMPTY'}")
                print(f"  token_id found: {token_id}")
                # Check clobTokenIds
                clob_ids = m.raw.get("clobTokenIds")
                print(f"  clobTokenIds: {clob_ids}")

            # Try clobTokenIds as fallback
            if not token_id:
                clob_ids = m.raw.get("clobTokenIds")
                if clob_ids:
                    try:
                        import json
                        ids = json.loads(clob_ids) if isinstance(clob_ids, str) else clob_ids
                        # Index 0 = YES, Index 1 = NO typically
                        if side == "YES" and len(ids) > 0:
                            token_id = ids[0]
                        elif side == "NO" and len(ids) > 1:
                            token_id = ids[1]
                    except:
                        pass

            high_prob.append({
                "question": m.question,
                "side": side,
                "price": high_p,
                "volume": m.volume or 0,
                "liquidity": m.liquidity or 0,
                "token_id": token_id,
                "category": m.category,
            })

    # Sort by price descending, then get actual spreads for 90-96% zone
    high_prob.sort(key=lambda x: x["price"], reverse=True)

    print("=" * 90)
    print("FETCHING ACTUAL ORDER BOOK SPREADS FOR 90-96% MARKETS")
    print("=" * 90)

    tradeable = []
    for m in high_prob:
        if 0.90 <= m["price"] < 0.96 and m["token_id"]:
            # Fetch actual order book
            book = await adapter.get_order_book(m["token_id"])

            if book.bids and book.asks:
                best_bid = book.bids[0].price
                best_ask = book.asks[0].price
                spread = best_ask - best_bid
                spread_pct = spread / ((best_bid + best_ask) / 2) * 100
                mid = (best_bid + best_ask) / 2
            else:
                spread_pct = None
                mid = m["price"]

            m["spread_pct"] = spread_pct
            m["mid"] = mid
            m["best_bid"] = best_bid if book.bids else None
            m["best_ask"] = best_ask if book.asks else None
            tradeable.append(m)

            # Small delay to avoid rate limiting
            await asyncio.sleep(0.1)

    await adapter.disconnect()

    # Print results with actual spread data
    print(f"\nFound {len(tradeable)} markets in 90-96% zone\n")

    print("=" * 90)
    print("ACTUAL SPREADS AND E(X) CALCULATION")
    print("=" * 90)
    print(f"\n{'Market':<45} {'Side':<4} {'Price':>6} {'Spread':>7} {'Net Edge':>8} {'E(X)/$25':>9}")
    print("-" * 90)

    # Assume 2% gross edge from research
    gross_edge = 0.02

    positive_ex = []
    for m in sorted(tradeable, key=lambda x: x["spread_pct"] if x["spread_pct"] else 999):
        spread = m["spread_pct"]
        if spread is not None:
            # Net edge = gross edge - spread (spread is cost)
            net_edge = gross_edge - (spread / 100)
            ex_25 = 25 * net_edge / m["price"]  # E(X) on $25 bet

            marker = "+" if net_edge > 0 else ""
            print(f"{m['question'][:45]:<45} {m['side']:<4} {m['price']:>5.1%} {spread:>6.2f}% {net_edge:>7.2%} ${ex_25:>7.2f}")

            if net_edge > 0:
                m["net_edge"] = net_edge
                m["ex_25"] = ex_25
                positive_ex.append(m)
        else:
            print(f"{m['question'][:45]:<45} {m['side']:<4} {m['price']:>5.1%} {'N/A':>7} {'N/A':>8} {'N/A':>9}")

    print("-" * 90)

    # Summary
    print(f"\n" + "=" * 90)
    print("MARKETS WITH POSITIVE E(X) (spread < 2% gross edge)")
    print("=" * 90)

    if positive_ex:
        total_ex = sum(m["ex_25"] for m in positive_ex)
        print(f"\n{len(positive_ex)} markets have positive E(X):\n")
        for m in positive_ex:
            print(f"  - {m['question'][:50]}")
            print(f"    {m['side']} @ {m['price']:.1%} | Spread: {m['spread_pct']:.2f}% | E(X): ${m['ex_25']:.2f}/trade")
            print(f"    Category: {m['category']}")
            print()

        print(f"Total E(X) if trading all {len(positive_ex)} markets: ${total_ex:.2f}")
        print(f"Total capital required: ${25 * len(positive_ex)}")
    else:
        print("\nNO markets have positive E(X) after spread costs!")

    # Category breakdown
    print(f"\n" + "=" * 90)
    print("CATEGORY BREAKDOWN (all 90-96% markets)")
    print("=" * 90)

    categories = {}
    for m in tradeable:
        cat = m["category"] or "unknown"
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(m)

    for cat, mkts in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"\n{cat}: {len(mkts)} markets")
        for m in mkts[:3]:
            spread_str = f"{m['spread_pct']:.2f}%" if m['spread_pct'] else "N/A"
            print(f"  - {m['question'][:50]} | {m['side']} @ {m['price']:.1%} | Spread: {spread_str}")

    return positive_ex


if __name__ == "__main__":
    asyncio.run(investigate())
