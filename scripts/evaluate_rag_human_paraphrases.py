import json

from evaluate_rag_retrieval import load_documents, retrieve


def find_docs(docs, *needles):
    matches = []
    for doc in docs:
        lowered = doc.lower()
        if all(needle.lower() in lowered for needle in needles):
            matches.append(doc)
    if not matches:
        raise RuntimeError(f"No document matched {needles}")
    return matches


def first_doc(docs, *needles):
    return find_docs(docs, *needles)[0]


def build_cases(docs):
    gold = {
        "premium": [first_doc(docs, "buyer's premium")],
        "reserve": [first_doc(docs, "reserve price")],
        "how_bid": [first_doc(docs, "how to bid")],
        "autobid": [first_doc(docs, "autobid proxy bidding")],
        "winning": [first_doc(docs, "winning an auction")],
        "sniping": [first_doc(docs, "sniping last minute bidding")],
        "conditions": [first_doc(docs, "item conditions")],
        "selling": [first_doc(docs, "selling on nexus")],
        "categories": [first_doc(docs, "categories:")],
        "ratings": [first_doc(docs, "bidder ratings")],
        "duration": [first_doc(docs, "auction duration comparison")],
        "cartier_stats": [first_doc(docs, "auction statistics for cartier")],
        "palm_stats": [first_doc(docs, "auction statistics for palm pilot")],
        "xbox_stats": [first_doc(docs, "auction statistics for xbox")],
        "cartier_pattern": [first_doc(docs, "bidding patterns for cartier")],
        "palm_pattern": [first_doc(docs, "bidding patterns for palm pilot")],
        "xbox_pattern": [first_doc(docs, "bidding patterns for xbox")],
        "cats": find_docs(docs, "cat"),
        "balloon": [first_doc(docs, "items starting with 'balloon'")],
        "teatime": [first_doc(docs, "items starting with 'teatime'")],
        "charlie_lola": find_docs(docs, "charlie", "lola"),
        "set4": [first_doc(docs, "items starting with 'set/4'")],
        "tlight": find_docs(docs, "t-light"),
        "redwhite": find_docs(docs, "red", "white"),
        "washing": find_docs(docs, "washing"),
    }

    return [
        ("premium_1", "how much extra do I pay after I win a lot?", gold["premium"]),
        ("premium_2", "is there an additional buyer fee?", gold["premium"]),
        ("premium_3", "if my winning bid is 100 what is the total?", gold["premium"]),
        ("premium_4", "what percent does Nexus add to the hammer price?", gold["premium"]),
        ("premium_5", "do you charge a buyer premium?", gold["premium"]),
        ("reserve_1", "what happens if nobody reaches the minimum seller price?", gold["reserve"]),
        ("reserve_2", "explain reserve met on an auction", gold["reserve"]),
        ("reserve_3", "can an item stay unsold after bidding?", gold["reserve"]),
        ("reserve_4", "what is the lowest amount a seller accepts called?", gold["reserve"]),
        ("reserve_5", "does the seller have a hidden minimum?", gold["reserve"]),
        ("bid_1", "where do I enter my maximum offer?", gold["how_bid"]),
        ("bid_2", "what button do I press to bid?", gold["how_bid"]),
        ("bid_3", "how does bidding work on Nexus?", gold["how_bid"]),
        ("autobid_1", "will the site bid automatically for me?", gold["autobid"]),
        ("autobid_2", "can other bidders see my maximum bid?", gold["autobid"]),
        ("autobid_3", "what is proxy bidding?", gold["autobid"]),
        ("winning_1", "what happens after I win?", gold["winning"]),
        ("winning_2", "how long do I have to pay after winning?", gold["winning"]),
        ("winning_3", "when will my item ship after payment?", gold["winning"]),
        ("sniping_1", "what does last second bidding mean?", gold["sniping"]),
        ("sniping_2", "how do I protect myself from snipers?", gold["sniping"]),
        ("sniping_3", "is bidding in the final seconds allowed?", gold["sniping"]),
        ("condition_1", "what does excellent condition mean?", gold["conditions"]),
        ("condition_2", "do expensive items include authenticity papers?", gold["conditions"]),
        ("condition_3", "difference between mint and fair condition", gold["conditions"]),
        ("selling_1", "how much commission do sellers pay?", gold["selling"]),
        ("selling_2", "does Nexus photograph my item if I sell?", gold["selling"]),
        ("selling_3", "when does a seller receive payment?", gold["selling"]),
        ("categories_1", "what kinds of products are listed?", gold["categories"]),
        ("categories_2", "do you sell watches and coins?", gold["categories"]),
        ("ratings_1", "what does bidder score mean?", gold["ratings"]),
        ("ratings_2", "how do I know if a buyer is reliable?", gold["ratings"]),
        ("cartier_stats_1", "typical closing amount for Cartier watches", gold["cartier_stats"]),
        ("cartier_stats_2", "what price range do Cartier wristwatches sell for?", gold["cartier_stats"]),
        ("cartier_stats_3", "how many bids were recorded for Cartier?", gold["cartier_stats"]),
        ("palm_stats_1", "average sale price for Palm M515 PDA", gold["palm_stats"]),
        ("palm_stats_2", "opening bid average for Palm Pilot", gold["palm_stats"]),
        ("palm_stats_3", "what is the price range for Palm Pilot M515?", gold["palm_stats"]),
        ("xbox_stats_1", "average Xbox auction closing price", gold["xbox_stats"]),
        ("xbox_stats_2", "how many bids for Xbox consoles?", gold["xbox_stats"]),
        ("xbox_stats_3", "Xbox price range in auctions", gold["xbox_stats"]),
        ("cartier_pattern_1", "when do people usually bid on Cartier watches?", gold["cartier_pattern"]),
        ("cartier_pattern_2", "Cartier sniping percentage near the end", gold["cartier_pattern"]),
        ("palm_pattern_1", "Palm PDA bidding activity in first day", gold["palm_pattern"]),
        ("palm_pattern_2", "Palm final 12 hour bid share", gold["palm_pattern"]),
        ("xbox_pattern_1", "Xbox console final 12 hour sniping rate", gold["xbox_pattern"]),
        ("xbox_pattern_2", "are Xbox bids concentrated near auction end?", gold["xbox_pattern"]),
        ("duration_1", "do seven day listings make more money?", gold["duration"]),
        ("duration_2", "compare 3 day and 5 day auctions", gold["duration"]),
        ("duration_3", "which duration attracts more bidders?", gold["duration"]),
        ("cats_1", "show me cat themed lots", gold["cats"]),
        ("cats_2", "anything with cats?", gold["cats"]),
        ("cats_3", "cute cats tape", gold["cats"]),
        ("balloon_1", "do you have balloon products?", gold["balloon"]),
        ("balloon_2", "balloon pump with 10 balloons", gold["balloon"]),
        ("teatime_1", "find teatime stationery", gold["teatime"]),
        ("teatime_2", "teatime gel pens", gold["teatime"]),
        ("charlie_1", "charlie lola hot water bottle", gold["charlie_lola"]),
        ("charlie_2", "items starting charlie+lola", gold["charlie_lola"]),
        ("set4_1", "set of 4 badges", gold["set4"]),
        ("set4_2", "set/4 items", gold["set4"]),
        ("tlight_1", "t light candle holder", gold["tlight"]),
        ("tlight_2", "t-light glass fluted antique", gold["tlight"]),
        ("redwhite_1", "red white dot mini cases", gold["redwhite"]),
        ("redwhite_2", "red/white picnic bag", gold["redwhite"]),
        ("washing_1", "washing item", gold["washing"]),
    ]


def evaluate():
    docs = load_documents()
    cases = build_cases(docs)
    ks = [1, 3, 5, 10]
    rows = []

    for case_id, query, gold_docs in cases:
        retrieved = retrieve(query, docs, max(ks))
        first_rank = None
        for index, doc in enumerate(retrieved, start=1):
            if doc in gold_docs:
                first_rank = index
                break

        row = {"id": case_id, "query": query, "first_rank": first_rank}
        for k in ks:
            row[f"recall@{k}"] = 1.0 if first_rank and first_rank <= k else 0.0
            row[f"mrr@{k}"] = (1.0 / first_rank) if first_rank and first_rank <= k else 0.0
        rows.append(row)

    summary = {
        "docs": len(docs),
        "n_cases": len(cases),
    }
    for k in ks:
        summary[f"recall@{k}"] = round(sum(row[f"recall@{k}"] for row in rows) / len(rows), 4)
        summary[f"mrr@{k}"] = round(sum(row[f"mrr@{k}"] for row in rows) / len(rows), 4)

    failures = [row for row in rows if row["recall@5"] == 0]
    summary["failures@5_count"] = len(failures)
    summary["failures@5_sample"] = failures[:20]
    return summary


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, ensure_ascii=False))
