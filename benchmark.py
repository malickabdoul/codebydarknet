"""
benchmark.py
-------------
Compare les backends de recherche sur trois tâches, avec le corpus comme
vérité terrain.

    python3 benchmark.py

Ce qu'on cherche à vérifier : est-ce que remplacer TF-IDF par des embeddings
serait une bonne idée, ou faut-il garder les deux ?

Limite à garder en tête : la vérité terrain vient du corpus lui-même, et
l'index est construit sur ce corpus. Ces chiffres mesurent la qualité de
*récupération*, pas la généralisation à des formulations inédites.
"""

import sys

import embeddings
import search
import translate


def backends_dispo():
    return list(search.BACKENDS) if embeddings.index_disponible() else ["tfidf"]


def tache_expressions(meta, backends, k=5):
    """Retrouver une expression mooré à partir de sa définition française."""
    exp = meta[meta["categorie_grammaticale"].astype(str)
               .str.contains("expression", case=False, na=False)]
    exp = exp[exp["definition_francais"].astype(str).str.split().str.len() >= 2]

    scores = {b: [0, 0] for b in backends}
    for _, row in exp.iterrows():
        attendu = search.nettoyer_mot_moore(str(row["mot_moore"]))
        q = str(row["definition_francais"])
        for b in backends:
            mots = [r["mot_moore"] for r in search.search(q, k=k, backend=b)]
            if mots and mots[0] == attendu:
                scores[b][0] += 1
            if attendu in mots:
                scores[b][1] += 1
    return scores, len(exp)


def _avec_faute(mot):
    """Retire un caractère au milieu, pour simuler une frappe approximative."""
    i = len(mot) // 2
    return mot[:i] + mot[i + 1:]


def tache_moore(meta, backends, k=5, n=300):
    """Retrouver une entrée mooré MAL ORTHOGRAPHIÉE.

    Interroger avec le mot exact ne mesurerait rien : search() fait passer les
    correspondances exactes devant, donc tous les backends obtiendraient 100 %
    sans que l'index intervienne. On introduit donc une faute de frappe, ce qui
    force le passage par les vecteurs — et correspond au cas réel d'un
    utilisateur qui ne connaît pas l'orthographe exacte.

    L'index sémantique n'encode que les définitions fr/en, le modèle ne sachant
    pas lire le mooré : il ne peut pas réussir ici. C'est l'argument contre le
    remplacement pur.
    """
    ech = meta.sample(n=min(n, len(meta)), random_state=42)
    scores = {b: [0, 0] for b in backends}
    total = 0
    for _, row in ech.iterrows():
        attendu = search.nettoyer_mot_moore(str(row["mot_moore"]))
        if len(attendu) < 4:
            continue  # trop court pour rester reconnaissable après la faute
        requete = _avec_faute(attendu)
        if search.positions_exactes(requete):
            continue  # la faute tombe par hasard sur une autre entrée
        total += 1
        for b in backends:
            mots = [r["mot_moore"] for r in search.search(requete, k=k, backend=b)]
            if mots and mots[0] == attendu:
                scores[b][0] += 1
            if attendu in mots:
                scores[b][1] += 1
    return scores, total


def tache_idiomes(backends):
    """Le bruit sur les phrases descriptives a-t-il disparu ?"""
    cas = [
        ("tu vas bien", "laafɩ bala", True),
        ("je veux boire de l'eau", None, False),
        ("comment allez-vous", None, False),
        ("merci beaucoup", None, False),
    ]
    lignes = []
    for phrase, attendu, doit_trouver in cas:
        for b in backends:
            if b == "hybride":
                continue  # scores RRF non interprétables comme proximité
            sugg = translate.suggestions_idiomatiques(phrase, k=3, backend=b)
            tete = f"{sugg[0]['mot_moore']} ({sugg[0]['score']:.2f})" if sugg else "—"
            ok = ""
            if doit_trouver and attendu:
                ok = " OK" if any(s["mot_moore"] == attendu for s in sugg) else " RATE"
            lignes.append((phrase, b, len(sugg), tete, ok))
    return lignes


def tableau(titre, scores, total, k):
    print(f"\n{titre}  ({total} cas)")
    print(f"  {'backend':14s} {'top-1':>8s} {'top-' + str(k):>8s}")
    for b, (t1, tk) in scores.items():
        print(f"  {b:14s} {t1 / total * 100:7.1f}% {tk / total * 100:7.1f}%")


def main():
    backends = backends_dispo()
    if len(backends) == 1:
        print("Index sémantique absent : seul « tfidf » est mesurable.")
        print("Pour comparer : python3 build_st_index.py\n")

    _, _, meta = search.load_all()
    k = 5

    print("=" * 62)
    print("BENCHMARK DES BACKENDS DE RECHERCHE")
    print("=" * 62)

    s, n = tache_expressions(meta, backends, k)
    tableau("1. Expression mooré retrouvée depuis sa définition française",
            s, n, k)

    s, n = tache_moore(meta, backends, k)
    tableau("2. Entrée mooré retrouvée malgré une faute de frappe", s, n, k)

    print("\n3. Tournures idiomatiques (bruit sur phrases descriptives)")
    print(f"  {'phrase':26s} {'backend':12s} {'n':>2s}  meilleure suggestion")
    for phrase, b, n_sugg, tete, ok in tache_idiomes(backends):
        print(f"  {phrase:26s} {b:12s} {n_sugg:2d}  {tete}{ok}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
