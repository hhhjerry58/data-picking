"""Document grouping, target overlap checks and loss windows for web retrieval."""
from collections import Counter, defaultdict
import re
from urllib.parse import urlsplit

from datasketch import MinHash, MinHashLSH

from .common import digest
from .data import normalize


def words(text):
    return re.findall(r"\w+|[^\w\s]", normalize(text))


def shingles(tokens, n=5):
    n = min(n, len(tokens))
    return {"\x1f".join(tokens[i:i+n]).encode() for i in range(len(tokens)-n+1)} if n else set()


def canonical_url(url):
    try:
        value = urlsplit(url)
        return (value.hostname or "").lower() + (value.path or "/") + ("?"+value.query if value.query else "")
    except ValueError:
        return url.strip()


def grouped_duplicates(rows, text_field, threshold, seed, url_field=None):
    """Exact identity plus MinHash candidates, verified by 5-token Jaccard.

    LSH is an approximate candidate search, not an exhaustive semantic check.
    All input rows are retained; callers decide how groups enter splits.
    """
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    lsh = MinHashLSH(threshold=threshold, num_perm=64, params=(16, 4))
    sets, exact, urls, edges = [], {}, {}, []
    for i, row in enumerate(rows):
        normalized = normalize(row[text_field])
        parts = shingles(words(normalized))
        sets.append(parts)
        key = digest(normalized)
        candidates = {}
        if key in exact: candidates[exact[key]] = "exact_text"
        exact.setdefault(key, i)
        if url_field and row.get(url_field):
            url = canonical_url(row[url_field])
            if url in urls: candidates[urls[url]] = "same_url"
            urls.setdefault(url, i)
        if parts:
            signature = MinHash(num_perm=64, seed=seed)
            signature.update_batch(sorted(parts))
            for other in lsh.query(signature): candidates.setdefault(int(other), "near_text")
            lsh.insert(str(i), signature)
        for j, kind in sorted(candidates.items()):
            similarity = len(parts & sets[j]) / max(1, len(parts | sets[j]))
            if kind != "near_text" or similarity >= threshold:
                union(i, j)
                edges.append({"a": rows[j]["id"], "b": row["id"], "kind": kind, "jaccard": similarity})
    groups = defaultdict(list)
    for i, row in enumerate(rows): groups[find(i)].append(row)
    for group in groups.values():
        key = digest(sorted(r["id"] for r in group))
        for row in group: row["group_id"] = key
    return list(groups.values()), edges


class TargetOverlap:
    """Detect exact/near containment of sufficiently distinctive target passages."""
    def __init__(self, targets, spec):
        self.spec = spec
        self.passages = []
        self.skipped_short = Counter()
        for row in targets:
            for field, minimum in [("prompt", spec["problem_min_tokens"]), ("response", spec["solution_min_tokens"])]:
                tokens = words(row[field])
                if len(tokens) < minimum:
                    self.skipped_short[field] += 1
                    continue
                self.passages.append({"id": row["id"], "field": field, "text": normalize(row[field]),
                                      "shingles": shingles(tokens)})
        frequency = Counter(p for row in self.passages for p in row["shingles"])
        self.index = defaultdict(set)
        for i, row in enumerate(self.passages):
            for part in sorted(row["shingles"], key=lambda p: (frequency[p], p))[:spec["anchors_per_text"]]:
                self.index[part].add(i)

    def matches(self, text):
        normalized = normalize(text)
        parts = shingles(words(normalized))
        candidates = set()
        for part in parts: candidates.update(self.index.get(part, ()))
        matches = []
        for i in sorted(candidates):
            row = self.passages[i]
            fraction = len(parts & row["shingles"]) / len(row["shingles"])
            exact = row["text"] in normalized
            if exact or fraction >= self.spec["containment_threshold"]:
                matches.append({"target_id": row["id"], "field": row["field"],
                                "exact_normalized_substring": exact, "shingle_containment": fraction})
        return matches


def document_windows(doc_id, token_ids, max_length):
    """Score each token after the document's first exactly once, with one-token overlap."""
    if max_length < 2 or len(token_ids) < 2:
        raise ValueError("A scoring window needs context and at least one predicted token")
    for index, start in enumerate(range(0, len(token_ids)-1, max_length-1)):
        ids = token_ids[start:start+max_length]
        mask = [0] + [1]*(len(ids)-1)
        yield {"id": f"{doc_id}:window:{index}", "document_id": doc_id, "window_index": index,
               "token_start": start, "token_end": start+len(ids), "input_ids": ids,
               "response_mask": mask, "scored_tokens": sum(mask),
               "tokenization_hash": digest({"input_ids": ids, "response_mask": mask})}
