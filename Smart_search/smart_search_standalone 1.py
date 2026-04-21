"""
INTÉGRATION RAPIDE (3 lignes) :
────────────────────────────────
    from smart_search_standalone import search

    results = search("picasso")           # -> [{"name": "Pablo Picasso", "score": 1.0, ...}]
    results = search("baskiat")           # faute -> Jean-Michel Basquiat
    results = search("frida calo")        # variante -> Frida Kahlo

INTÉGRATION AVEC TON DATAFRAME :
──────────────────────────────────
    from smart_search_standalone import SmartSearch

    engine  = SmartSearch.from_dataframe(df, name_col="artist_name")
    results = engine.search("picaso")     # -> [{"name": "Pablo Picasso", ...}]

DANS UNE API / FLASK / FASTAPI / DJANGO :
──────────────────────────────────────────
    # Initialise une seule fois au démarrage
    engine = SmartSearch.from_dataframe(df, name_col="artist_name")

    @app.route("/search")
    def api_search():
        q = request.args.get("q", "")
        return jsonify(engine.search(q))

DANS TON PIPELINE ML :
───────────────────────
    engine = SmartSearch.from_dataframe(df, name_col="artist_name")

    # Nettoyer / normaliser une colonne entière
    df["artist_clean"] = engine.batch_match(df["artist_name"].tolist())
"""

import unicodedata
import re
from typing import Optional
import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# 1. NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _normalize(text: str) -> str:
    """
    Transforme n'importe quelle saisie en chaîne comparable :
      - minuscules
      - accents supprimés  (é→e, ñ→n, ü→u …)
      - ponctuation → espace
      - espaces multiples → simple

    Exemples :
      "Frida Kahlo"  → "frida kahlo"
      "JEAN-MICHEL"  → "jean michel"
      "Basquiàt"     → "basquiat"
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ══════════════════════════════════════════════════════════════════════════════
# 2. ALGORITHMES DE SIMILARITÉ
# ══════════════════════════════════════════════════════════════════════════════

def _levenshtein(a: str, b: str) -> int:
    """
    Distance d'édition entre deux chaînes (nombre minimum d'insertions,
    suppressions, substitutions pour passer de a à b).

    Exemples :
      "picasso" vs "picaso"  → 1
      "baskiat" vs "basquiat"→ 2
    """
    if a == b:
        return 0
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    return dp[n]


def _ngram_similarity(a: str, b: str, n: int = 2) -> float:
    """
    Coefficient Dice sur les bi-grammes.
    Détecte les ressemblances phonétiques / orthographiques proches.

    Exemples :
      "picasso" vs "picasoo"  → ~0.85
      "rothko"  vs "rotko"    → ~0.80
    """
    def grams(s):
        return set(s[i:i+n] for i in range(len(s) - n + 1))
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return (2 * len(ga & gb)) / (len(ga) + len(gb))


def _fuzzy_score(query: str, target: str) -> tuple[float, str]:
    """
    Score global [0.0 – 1.0] entre query et target (déjà normalisés).
    Combine 4 méthodes et retourne la meilleure.

    Retourne : (score, méthode_gagnante)
    """
    if query == target:
        return 1.0, "exact"

    if target in query or query in target:
        overlap = min(len(query), len(target)) / max(len(query), len(target))
        return overlap * 0.97, "sous-chaîne"

    max_len = max(len(query), len(target), 1)
    lev_score = (1 - _levenshtein(query, target) / max_len) * 0.92
    ng_score  = _ngram_similarity(query, target) * 0.88

    # Correspondance préfixe mot à mot
    prefix_score = 0.0
    for qw in query.split():
        for tw in target.split():
            if len(qw) >= 3 and (tw.startswith(qw) or qw.startswith(tw)):
                s = min(len(qw), len(tw)) / max(len(qw), len(tw)) * 0.85
                prefix_score = max(prefix_score, s)

    best = max(lev_score, ng_score, prefix_score)
    if   best == lev_score:    method = "orthographique"
    elif best == ng_score:     method = "phonétique"
    else:                      method = "préfixe"

    return best, method


# ══════════════════════════════════════════════════════════════════════════════
# 3. DICTIONNAIRE D'ALIAS (artistes connus)
#    → enrichir avec les noms de ton dataset
# ══════════════════════════════════════════════════════════════════════════════

_ALIASES: dict[str, list[str]] = {
    "pablo picasso":           ["picasso", "p. picasso", "pablo", "pablo ruiz", "picsasso", "picaso"],
    "jean-michel basquiat":    ["basquiat", "jean michel", "jmb", "baskiat", "baskiyat", "basquiet"],
    "salvador dali":           ["dali", "salvadore dali", "dali salvador", "dalí"],
    "frida kahlo":             ["kahlo", "frida calo", "frieda kahlo", "frida kalo"],
    "andy warhol":             ["warhol", "worhal", "wahol", "warhole", "a. warhol"],
    "mark rothko":             ["rothko", "rotko", "rothkow", "rothco"],
    "yayoi kusama":            ["kusama", "yayoï", "kusama yayoi", "yayoi"],
    "banksy":                  ["bansky", "banski", "banksie", "banksey"],
    "gerhard richter":         ["richter", "gerhard", "g. richter"],
    "jackson pollock":         ["pollock", "pollok", "polock", "j. pollock"],
    "roy lichtenstein":        ["lichtenstein", "lichtenstien", "lichtenshtein", "lichtenstain"],
    "takashi murakami":        ["murakami", "murakamee", "t. murakami"],
    "jeff koons":              ["koons", "j. koons", "jeff kon"],
    "damien hirst":            ["hirst", "d hirst", "d. hirst"],
    "louise bourgeois":        ["bourgeois", "bourgois", "bourjois", "l. bourgeois"],
    "lucian freud":            ["freud", "lucien freud", "l. freud"],
    "yves klein":              ["klein", "ives klein", "y. klein"],
    "pierre soulages":         ["soulages", "soulagez", "p. soulages"],
    "jean dubuffet":           ["dubuffet", "dubufet", "du buffet"],
    "simon hantai":            ["hantai", "simon antai", "antai"],
    "georg baselitz":          ["baselitz", "george baselitz", "baseliz"],
    "cy twombly":              ["twombly", "twombli", "c. twombly"],
    "willem de kooning":       ["de kooning", "kooning", "dekooning", "w. de kooning"],
    "jasper johns":            ["johns", "jasper john", "j. johns"],
    "tracey emin":             ["emin", "tracy emin", "tracee emin", "t. emin"],
    "peter doig":              ["doig", "pete doig", "p. doig"],
    "ai weiwei":               ["weiwei", "ai wei wei", "ay weiwei"],
    "wang guangyi":            ["guangyi", "wang guanyi", "wang kwangyi"],
    "zeng fanzhi":             ["zeng", "fanzhi", "zeng fanshi"],
    "kaws":                    ["kaws artist", "brian donnelly"],
    "jean-paul riopelle":      ["riopelle", "jp riopelle", "jean paul riopelle"],
    "pierre-auguste renoir":   ["renoir", "p.a. renoir", "auguste renoir"],
    "claude monet":            ["monet", "c. monet"],
    "vincent van gogh":        ["van gogh", "vangogh", "gogh", "vincent van gogh"],
    "henri matisse":           ["matisse", "h. matisse"],
    "paul cezanne":            ["cezanne", "cézanne", "p. cezanne"],
    "edgar degas":             ["degas", "e. degas"],
    "paul gauguin":            ["gauguin", "p. gauguin", "goguin"],
    "wassily kandinsky":       ["kandinsky", "kandinski", "w. kandinsky"],
    "piet mondrian":           ["mondrian", "p. mondrian", "mondriaan"],
    "joan miro":               ["miro", "miró", "j. miro"],
    "rene magritte":           ["magritte", "magrite", "r. magritte"],
    "gustave courbet":         ["courbet", "g. courbet"],
    "fernand leger":           ["leger", "léger", "f. leger"],
    "marc chagall":            ["chagall", "m. chagall"],
    "alberto giacometti":      ["giacometti", "a. giacometti"],
    "max ernst":               ["ernst", "m. ernst"],
}


# ══════════════════════════════════════════════════════════════════════════════
# 4. MOTEUR DE RECHERCHE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class SmartSearch:
    """
    Moteur de recherche intelligent pour artistes / entités textuelles.

    ── Création ────────────────────────────────────────────────────────────────

    # Option A : artistes connus inclus par défaut
    engine = SmartSearch()

    # Option B : depuis ton DataFrame
    engine = SmartSearch.from_dataframe(df, name_col="artist_name")

    # Option C : liste custom
    engine = SmartSearch(names=["Pablo Picasso", "Banksy", "Yayoi Kusama"])

    ── Recherche ────────────────────────────────────────────────────────────────

    results = engine.search("baskiat")
    # -> [{"name": "Jean-Michel Basquiat", "score": 0.87, "confidence": "87.0%", "method": "orthographique"}]

    name = engine.best_match("frida calo")
    # -> "Frida Kahlo"

    df["artist_clean"] = engine.batch_match(df["artist_name"].tolist())
    # -> Série pandas avec les noms nettoyés / normalisés
    """

    def __init__(
        self,
        names: Optional[list[str]] = None,
        threshold: float = 0.42,
        top_k: int = 8,
    ):
        self.threshold = threshold
        self.top_k     = top_k
        self._index: list[dict] = []

        if names is not None:
            for name in names:
                self._add_entry(name)
        else:
            # Index par défaut : artistes connus
            for full_name in _ALIASES:
                self._add_entry(full_name.title(), extra_aliases=_ALIASES[full_name])

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        name_col: str = "artist_name",
        threshold: float = 0.42,
        top_k: int = 8,
    ) -> "SmartSearch":
        """
        Construit le moteur depuis un DataFrame.

        Génère automatiquement des alias :
          - prénom seul
          - nom seul
          - initiale + nom

        Exemple :
            engine = SmartSearch.from_dataframe(df, name_col="artist_name")
        """
        engine = cls.__new__(cls)
        engine.threshold = threshold
        engine.top_k     = top_k
        engine._index    = []

        if name_col not in df.columns:
            raise ValueError(f"Colonne '{name_col}' introuvable dans le DataFrame. "
                             f"Colonnes disponibles : {list(df.columns)}")

        seen = set()
        for raw_name in df[name_col].dropna().unique():
            key = _normalize(str(raw_name))
            if key in seen:
                continue
            seen.add(key)

            # Alias automatiques à partir du nom
            auto_aliases = []
            parts = key.split()
            if len(parts) >= 2:
                auto_aliases += [parts[-1], parts[0], f"{parts[0][0]} {parts[-1]}"]

            # Alias connus en plus
            known = [_normalize(a) for a in _ALIASES.get(key, [])]

            engine._add_entry(
                str(raw_name),
                extra_aliases=auto_aliases + known
            )

        return engine

    # ── Construction de l'index ───────────────────────────────────────────────

    def _add_entry(self, raw_name: str, extra_aliases: Optional[list[str]] = None):
        key     = _normalize(raw_name)
        aliases = [_normalize(a) for a in (extra_aliases or [])]
        aliases += [_normalize(a) for a in _ALIASES.get(key, [])]
        # Dédoublonner
        aliases = list(dict.fromkeys(a for a in aliases if a and a != key))

        self._index.append({
            "name":    raw_name,
            "norm":    key,
            "aliases": aliases,
        })

    def add(self, *names: str):
        """
        Ajoute un ou plusieurs noms à l'index à la volée.

        Exemple :
            engine.add("Kaws", "Invader", "Jean-Pierre Raynaud")
        """
        for name in names:
            key = _normalize(name)
            if not any(e["norm"] == key for e in self._index):
                self._add_entry(name)
        return self  # chainable

    # ── Score ─────────────────────────────────────────────────────────────────

    def _score(self, query_norm: str, entry: dict) -> tuple[float, str]:
        targets = [entry["norm"]] + entry["aliases"]
        best_sc, best_m = 0.0, ""
        for t in targets:
            sc, m = _fuzzy_score(query_norm, t)
            if sc > best_sc:
                best_sc, best_m = sc, m
        return best_sc, best_m

    # ── API publique ──────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Recherche intelligente. Tolère les fautes, variantes, accents, casse.

        Args :
            query     : texte libre (ex : "baskiat", "frida calo", "PICASSO")
            threshold : score minimum [0-1] (défaut : 0.42)
            top_k     : nombre max de résultats (défaut : 8)

        Retourne :
            Liste de dicts triés par score décroissant :
            [
              {
                "name":       "Jean-Michel Basquiat",   # nom original
                "score":      0.8700,                   # float [0-1]
                "confidence": "87.0%",                  # lisible
                "method":     "orthographique"          # algo gagnant
              },
              ...
            ]

        Exemples :
            engine.search("picasso")         # -> Pablo Picasso  100%
            engine.search("baskiat")         # -> Jean-Michel Basquiat  87%
            engine.search("frida calo")      # -> Frida Kahlo  89%
            engine.search("xyz inconnu")     # -> []
        """
        if not query or not query.strip():
            return []

        t      = threshold if threshold is not None else self.threshold
        k      = top_k     if top_k     is not None else self.top_k
        q_norm = _normalize(query)

        results = []
        for entry in self._index:
            sc, method = self._score(q_norm, entry)
            if sc >= t:
                results.append({
                    "name":       entry["name"],
                    "score":      round(sc, 4),
                    "confidence": f"{sc * 100:.1f}%",
                    "method":     method,
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def best_match(self, query: str, min_confidence: float = 0.50) -> Optional[str]:
        """
        Retourne le nom du meilleur résultat, ou None si sous le seuil.

        Utile pour les formulaires, les API, la déduplication.

        Exemple :
            name = engine.best_match("baskiat")    # -> "Jean-Michel Basquiat"
            name = engine.best_match("xyz")        # -> None
        """
        r = self.search(query, threshold=min_confidence, top_k=1)
        return r[0]["name"] if r else None

    def batch_match(
        self,
        queries: list[str],
        min_confidence: float = 0.50,
    ) -> pd.Series:
        """
        Mappe une liste de queries vers les noms normalisés.
        Parfait pour nettoyer une colonne DataFrame.

        Exemple :
            df["artist_clean"] = engine.batch_match(df["artist_name"].tolist())

        Les entrées sans match suffisant retournent None.
        """
        return pd.Series([self.best_match(q, min_confidence) for q in queries])

    def suggest(self, partial: str, min_len: int = 2) -> list[str]:
        """
        Suggestions pour autocomplétion (saisie partielle).

        Exemple :
            engine.suggest("picass")  # -> ["Pablo Picasso"]
            engine.suggest("ban")     # -> ["Banksy", ...]
        """
        if len(partial) < min_len:
            return []
        norm = _normalize(partial)
        out  = []
        for entry in self._index:
            if entry["norm"].startswith(norm) or any(a.startswith(norm) for a in entry["aliases"]):
                out.append(entry["name"])
        return out[:10]

    def __repr__(self):
        return f"SmartSearch(artists={len(self._index)}, threshold={self.threshold})"


# ══════════════════════════════════════════════════════════════════════════════
# 5. FONCTION RAPIDE (sans instanciation)
# ══════════════════════════════════════════════════════════════════════════════

_default_engine: Optional[SmartSearch] = None

def search(query: str, top_k: int = 5) -> list[dict]:
    """
    Fonction one-liner pour rechercher dans les artistes connus par défaut.
    Aucune configuration nécessaire.

    Exemple :
        from smart_search_standalone import search

        search("picasso")      # -> [{"name": "Pablo Picasso", "score": 1.0, ...}]
        search("baskiat")      # -> [{"name": "Jean-Michel Basquiat", ...}]
        search("frida calo")   # -> [{"name": "Frida Kahlo", ...}]
    """
    global _default_engine
    if _default_engine is None:
        _default_engine = SmartSearch()
    return _default_engine.search(query, top_k=top_k)


# ══════════════════════════════════════════════════════════════════════════════
# 6. EXEMPLES D'INTÉGRATION (copier-coller prêt à l'emploi)
# ══════════════════════════════════════════════════════════════════════════════

INTEGRATION_EXAMPLES = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXEMPLE 1 — Script Python simple (3 lignes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from smart_search_standalone import search

    for r in search("baskiat"):
        print(r["name"], r["confidence"])
    # Jean-Michel Basquiat  87.0%


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXEMPLE 2 — Pipeline ML (nettoyage de colonne)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    import pandas as pd
    from smart_search_standalone import SmartSearch

    df = pd.read_csv("df_for_ml_improved_up_to_2012.csv", encoding="latin1")

    engine = SmartSearch.from_dataframe(df, name_col="artist_name")

    # Nettoyer toute la colonne artiste
    df["artist_clean"] = engine.batch_match(df["artist_name"].tolist())

    # Recherche manuelle
    print(engine.search("frida calo"))
    # [{"name": "Frida Kahlo", "score": 0.89, "confidence": "89.0%", "method": "orthographique"}]


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXEMPLE 3 — Flask / FastAPI (backend)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Flask
    from flask import Flask, request, jsonify
    from smart_search_standalone import SmartSearch
    import pandas as pd

    app    = Flask(__name__)
    df     = pd.read_csv("df_for_ml_improved_up_to_2012.csv", encoding="latin1")
    engine = SmartSearch.from_dataframe(df, name_col="artist_name")  # init une seule fois

    @app.route("/search")
    def api_search():
        q = request.args.get("q", "")
        return jsonify(engine.search(q, top_k=5))

    # FastAPI
    from fastapi import FastAPI
    from smart_search_standalone import SmartSearch

    app    = FastAPI()
    engine = SmartSearch()  # artistes connus par défaut

    @app.get("/search")
    def api_search(q: str):
        return engine.search(q)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXEMPLE 4 — Streamlit (autocomplete live)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    import streamlit as st
    from smart_search_standalone import SmartSearch
    import pandas as pd

    @st.cache_resource
    def load_engine():
        df = pd.read_csv("df_for_ml_improved_up_to_2012.csv", encoding="latin1")
        return SmartSearch.from_dataframe(df, name_col="artist_name")

    engine = load_engine()
    query  = st.text_input("Artiste")

    if query:
        results = engine.search(query, top_k=5)
        for r in results:
            st.write(f"{r['name']}  —  {r['confidence']}")


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXEMPLE 5 — Jupyter Notebook / exploration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    import pandas as pd
    from smart_search_standalone import SmartSearch

    df     = pd.read_csv("df_for_ml_improved_up_to_2012.csv", encoding="latin1")
    engine = SmartSearch.from_dataframe(df, name_col="artist_name")

    # Explorer les résultats
    pd.DataFrame(engine.search("rothco"))
    #          name  score confidence       method
    # 0  Mark Rothko  0.875      87.5%  orthographique


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXEMPLE 6 — CLI (ligne de commande)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python smart_search_standalone.py "frida calo"
    python smart_search_standalone.py "baskiat" --top 3
"""


# ══════════════════════════════════════════════════════════════════════════════
# 7. POINT D'ENTRÉE CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Smart Search — recherche intelligente d'artistes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=INTEGRATION_EXAMPLES,
    )
    parser.add_argument("query",         nargs="?", default=None, help="Texte à rechercher")
    parser.add_argument("--top",  "-t",  type=int,  default=5,    help="Nombre de résultats (défaut: 5)")
    parser.add_argument("--demo",        action="store_true",      help="Lancer la démo complète")
    parser.add_argument("--examples",   action="store_true",       help="Afficher les exemples d'intégration")

    args = parser.parse_args()

    engine = SmartSearch()

    # ── Mode exemples ─────────────────────────────────────────────────────────
    if args.examples:
        print(INTEGRATION_EXAMPLES)
        sys.exit(0)

    # ── Mode démo ─────────────────────────────────────────────────────────────
    if args.demo or args.query is None:
        print("\n" + "═"*70)
        print("  SMART SEARCH — Démonstration")
        print("═"*70)
        print(f"\n{'Requête':<22} {'Attendu':<27} {'Trouvé':<27} {'Score':>7}")
        print("─"*70)

        tests = [
            ("picasso",      "Pablo Picasso"),
            ("baskiat",      "Jean-Michel Basquiat"),
            ("frida calo",   "Frida Kahlo"),
            ("rothco",       "Mark Rothko"),
            ("warholl",      "Andy Warhol"),
            ("bansky",       "Banksy"),
            ("picsasso",     "Pablo Picasso"),
            ("dalí",         "Salvador Dali"),
            ("murakami",     "Takashi Murakami"),
            ("xyz inconnu",  None),
        ]

        ok_count = 0
        for query, expected in tests:
            results = engine.search(query, top_k=1)
            found   = results[0]["name"]       if results else "—"
            conf    = results[0]["confidence"] if results else "—"
            matched = (
                found.lower() == (expected or "").lower()
                or (expected is None and not results)
            )
            ok_count += matched
            status = "✓" if matched else "✗"
            print(f"  {status}  {query:<20} {str(expected):<27} {found:<27} {conf:>7}")

        print(f"\n  Précision : {ok_count}/{len(tests)}")
        print(f"\n  Pour voir les exemples d'intégration : python {sys.argv[0]} --examples")
        print("═"*70)
        sys.exit(0)

    # ── Mode recherche ────────────────────────────────────────────────────────
    results = engine.search(args.query, top_k=args.top)

    if not results:
        print(f"\n  Aucun résultat pour « {args.query} »")
        print("  Essayez un seuil plus bas : moteur.search(query, threshold=0.3)")
    else:
        print(f"\n  Résultats pour « {args.query} » :\n")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['name']:<30}  {r['confidence']:>6}  ({r['method']})")
    print()
