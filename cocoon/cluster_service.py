def compute_cluster(df, match: str = "matches"):
    clusters = {}

    for idx, row in df.iterrows():
        entry = row[match]
        if "similar_to" not in entry:
            clusters[idx] = []

    for idx, row in df.iterrows():
        entry = row[match]
        if "similar_to" in entry:
            similar_to_idx = entry["similar_to"]
            if similar_to_idx in clusters:
                clusters[similar_to_idx].append(idx)
            elif similar_to_idx not in clusters:
                clusters[similar_to_idx] = [idx]
    return clusters
