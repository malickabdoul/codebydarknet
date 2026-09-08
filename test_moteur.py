"""
test_moteur.py
---------------
Tests de non-régression.

    python3 test_moteur.py            # tout
    python3 test_moteur.py -v         # detaillé
    python3 -m unittest test_moteur   # equivalent

Pas de dépendance supplémentaire : `unittest` est dans la bibliothèque
standard. Les tests qui exigent l'index sémantique se sautent tout seuls s'il
n'a pas été construit.

Ce qui est testé ici, ce sont les endroits où le corpus piège : découpage des
domaines composés, en-têtes corrompus, colonnes synonymes/antonymes
mélangées, élisions et ligatures françaises, fragments de définition pris
pour des gloses. Chaque cas vient d'un vrai défaut rencontré dans les données,
pas d'un exemple inventé.
"""

import unittest

import embeddings
import liens
import recherche_web
import search
import translate

SEMANTIQUE = embeddings.index_disponible()
besoin_semantique = unittest.skipUnless(
    SEMANTIQUE, "index sémantique absent (python3 build_st_index.py)")


class TestDomaines(unittest.TestCase):
    """La colonne domaine mêle valeurs simples et composées, et certains
    domaines contiennent eux-mêmes une virgule."""

    def test_domaine_simple(self):
        self.assertEqual(search.split_domaines("Weather"), ["Weather"])

    def test_domaine_compose(self):
        self.assertEqual(search.split_domaines("Agriculture, Water"),
                         ["Agriculture", "Water"])

    def test_domaine_a_virgule_reste_entier(self):
        # « Bush, shrub » est UN domaine, pas deux.
        self.assertEqual(search.split_domaines("Bush, shrub"), ["Bush, shrub"])
        self.assertEqual(search.split_domaines("Food, Bush, shrub"),
                         ["Food", "Bush, shrub"])
        self.assertEqual(search.split_domaines("Agriculture, Grass, herb, vine"),
                         ["Agriculture", "Grass, herb, vine"])

    def test_toutes_les_valeurs_du_corpus_se_reconstruisent(self):
        _, _, meta = search.load_all()
        for valeur in set(meta["domaine"].dropna().astype(str)):
            self.assertEqual(", ".join(search.split_domaines(valeur)), valeur,
                             f"découpage non réversible : {valeur!r}")

    def test_liste_des_domaines_non_vide(self):
        doms = search.list_domaines()
        self.assertGreater(len(doms), 40)
        self.assertIn("Water", doms)
        self.assertIn("Bush, shrub", doms)


class TestEntetesCorrompus(unittest.TestCase):
    """Sept en-têtes du corpus mélangent le mot mooré et sa traduction."""

    def test_texte_etranger_devant(self):
        self.assertEqual(search.nettoyer_mot_moore("Come! y"), "y")
        self.assertEqual(search.nettoyer_mot_moore("why? bõeedga"), "bõeedga")
        self.assertEqual(
            search.nettoyer_mot_moore("Ne y waoongo ! Bienvenues ! Welcome! waoongo"),
            "waoongo")

    def test_moore_en_premier_quand_la_chaine_finit_par_ponctuation(self):
        self.assertEqual(
            search.nettoyer_mot_moore("Ges neere! Regarde bien ! Watch well!"),
            "Ges neere")

    def test_notations_legitimes_intactes(self):
        # Crochets de ton et parenthèses grammaticales ne sont pas des erreurs.
        for mot in ("[ã̀] ãbga", "ka be (sẽn)", "gepeyɛse (GPS)", "koom"):
            self.assertEqual(search.nettoyer_mot_moore(mot), mot)


class TestRenvois(unittest.TestCase):
    """Les colonnes synonymes et antonymes se contaminent mutuellement."""

    def test_chiffre_de_sens_retire(self):
        self.assertEqual(search._liste("wãbre, yãbre1, warpusi"),
                         ["wãbre", "yãbre", "warpusi"])

    def test_antonyme_cache_dans_la_colonne_synonymes(self):
        syn, ant = search.renvois("saamba; antonyme: ma1", "")
        self.assertEqual(syn, ["saamba"])
        self.assertEqual(ant, ["ma"])

    def test_synonyme_cache_dans_la_colonne_antonymes(self):
        syn, ant = search.renvois("", "kãsre, kãoogo4; synonyme: bɩgemde")
        self.assertEqual(ant, ["kãsre", "kãoogo"])
        self.assertEqual(syn, ["bɩgemde"])

    def test_valeurs_vides(self):
        self.assertEqual(search.renvois("", ""), ([], []))
        self.assertEqual(search.renvois(None, float("nan")), ([], []))


class TestDefinitionsTronquees(unittest.TestCase):
    """74 définitions sont coupées net à la source."""

    def test_parenthese_non_fermee_signalee(self):
        self.assertEqual(search.marquer_tronquee("têtu (lit"), "têtu (lit […]")
        self.assertEqual(search.marquer_tronquee("tomber (ex"), "tomber (ex […]")

    def test_definition_normale_intacte(self):
        for d in ("eau", "eau fraîche, eau froide", "ou bien (indique un choix)"):
            self.assertEqual(search.marquer_tronquee(d), d)


class TestCorrespondanceExacte(unittest.TestCase):
    """Chercher un mot du dictionnaire doit rendre ce mot en premier."""

    def test_mot_exact_en_tete(self):
        for mot in ("koom", "doogo", "bãg-tẽoko"):
            resultats = search.search(mot, k=3)
            self.assertEqual(resultats[0]["mot_moore"], mot)

    def test_sans_diacritiques(self):
        # Tout le monde n'a pas ã ou ʋ sur son clavier.
        resultats = search.search("bag-teoko", k=3)
        self.assertEqual(resultats[0]["mot_moore"], "bãg-tẽoko")

    def test_mot_inexistant_ne_leve_pas(self):
        self.assertEqual(search.positions_exactes("zzzqxwv"), [])


class TestRoutage(unittest.TestCase):
    """« auto » choisit l'index selon la langue de la requête."""

    def test_caracteres_moore(self):
        self.assertEqual(search.detecter_langue("bãg-tẽoko"), "moore")
        self.assertEqual(search.detecter_langue("kõo"), "moore")

    def test_mots_francais(self):
        for mot in ("eau", "maison", "manger"):
            self.assertEqual(search.detecter_langue(mot), "fr_en")

    def test_mot_inconnu_traite_comme_moore(self):
        # Un mot absent des deux vocabulaires est le plus souvent une entrée
        # mooré mal orthographiée : seul l'index lexical peut aider.
        self.assertEqual(search.detecter_langue("monre"), "moore")

    def test_moore_route_vers_entrees(self):
        self.assertEqual(search.resoudre_backend("auto", "koom"), "entrees")

    def test_backend_explicite_non_modifie(self):
        self.assertEqual(search.resoudre_backend("tfidf", "koom"), "tfidf")

    def test_backend_inconnu_leve(self):
        with self.assertRaises(ValueError):
            search.search("eau", backend="nawak")

    @besoin_semantique
    def test_francais_route_vers_hybride(self):
        self.assertEqual(search.resoudre_backend("auto", "eau"), "hybride")


class TestBornesRecherche(unittest.TestCase):
    """k a déjà fait tomber l'API en 500."""

    def test_k_negatif_ou_nul(self):
        for k in (0, -1, -100):
            self.assertEqual(len(search.search("eau", k=k)), 1)

    def test_k_superieur_au_corpus(self):
        index, _, _ = search.load_all()
        self.assertLessEqual(len(search.search("eau", k=99999)), index.ntotal)

    def test_champs_presents(self):
        r = search.search("ba", k=1)[0]
        for champ in ("mot_moore", "definition_francais", "categorie", "domaine",
                      "synonymes", "antonymes", "variantes", "pluriel",
                      "autres_sens", "score", "backend", "langue_detectee"):
            self.assertIn(champ, r)


class TestNormalisationFrancaise(unittest.TestCase):
    """Élisions et ligatures ont chacune causé un bug réel."""

    def test_elision_coupee(self):
        self.assertEqual(translate.normaliser("l'eau"), "l eau")
        self.assertEqual(translate.normaliser("qu'il"), "qu il")

    def test_ligature_preservee(self):
        # Sans conversion, « œil » perdait son œ et devenait « il ».
        self.assertEqual(translate.normaliser("œil"), "oeil")
        self.assertNotEqual(translate.normaliser("œil"), "il")

    def test_accents_conserves_dans_la_variante_accentuee(self):
        self.assertEqual(translate.normaliser_accents("marché"), "marché")
        self.assertEqual(translate.normaliser("marché"), "marche")


class TestGloses(unittest.TestCase):
    """L'index inverse français ne doit pas indexer des bouts de phrase."""

    def test_fragment_de_definition_rejete(self):
        # « ...griller la viande, le poisson » ne fait pas de « le poisson »
        # une glose autonome.
        _, gloses_acc, _ = translate._index_gloses()
        self.assertNotIn("le poisson", gloses_acc)
        self.assertNotIn("la viande", gloses_acc)

    def test_premier_fragment_toujours_garde(self):
        self.assertTrue(translate._glose_valide("à voix basse", 0))
        self.assertFalse(translate._glose_valide("le poisson", 1))
        self.assertTrue(translate._glose_valide("consommer", 1))


class TestSegmentation(unittest.TestCase):
    """Le cœur du mode traduction."""

    def _tetes(self, phrase):
        return [s["candidats"][0]["mot_moore"] if s["candidats"] else None
                for s in translate.segmenter(phrase)]

    def test_expression_reconnue_d_un_bloc(self):
        segments = translate.segmenter("n'importe comment")
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["type"], "expression")
        self.assertEqual(segments[0]["candidats"][0]["mot_moore"], "a bal")

    def test_forme_conjuguee_lemmatisee(self):
        segments = {s["source"]: s for s in translate.segmenter("je veux manger")}
        self.assertEqual(segments["veux"].get("lemme"), "vouloir")
        self.assertEqual(segments["veux"]["confiance"], "lemmatisé")

    def test_article_non_traduit(self):
        # « la » ne doit pas sortir « ka », qui est la négation.
        segments = {s["source"]: s for s in translate.segmenter("la maison")}
        self.assertEqual(segments["la"]["type"], "ignore")
        self.assertEqual(segments["la"]["candidats"], [])

    def test_accent_desambigue(self):
        self.assertIn("raaga", self._tetes("le marché"))     # le lieu
        self.assertIn("kẽnde", self._tetes("il marche"))     # l'action

    def test_phrase_entiere(self):
        res = translate.traduire("je veux boire de l'eau")
        self.assertEqual(res["couverture"], 1.0)
        self.assertIn("koom", res["glose"])
        self.assertIn("avertissement", res)

    def test_mot_inconnu_signale(self):
        segments = translate.segmenter("zzzqxwv")
        self.assertEqual(segments[0]["type"], "inconnu")


class TestIdiomes(unittest.TestCase):
    def test_phrase_trop_courte(self):
        self.assertEqual(translate.suggestions_idiomatiques("eau"), [])

    def test_tournure_trouvee(self):
        # « tu vas bien » se dit « laafɩ bala » : aucun mot ne coïncide.
        sugg = translate.suggestions_idiomatiques("tu vas bien", k=3)
        self.assertTrue(sugg)
        self.assertEqual(sugg[0]["mot_moore"], "laafɩ bala")

    @besoin_semantique
    def test_phrase_descriptive_sans_bruit(self):
        # Le veto sémantique doit écarter les faux positifs lexicaux.
        self.assertEqual(translate.suggestions_idiomatiques("la maison de mon père"), [])


class TestLiens(unittest.TestCase):
    def test_liens_produits(self):
        resultat = liens.liens_externes("koom", "eau")
        self.assertGreaterEqual(len(resultat), 5)
        for lien in resultat:
            self.assertTrue(lien["url"].startswith("https://"))
            for champ in ("nom", "url", "description", "source"):
                self.assertIn(champ, lien)

    def test_terme_encode(self):
        # Un mot à diacritiques ne doit pas casser l'URL.
        url = liens.liens_externes("bãg-tẽoko")[0]["url"]
        self.assertNotIn(" ", url)
        self.assertIn("%", url)

    def test_sources_douteuses_signalees(self):
        notes = [l for l in liens.liens_externes("koom") if "note" in l]
        self.assertEqual(len(notes), 2)  # Glosbe (404) et Webonary (Cloudflare)

    def test_terme_vide(self):
        self.assertEqual(liens.liens_externes(""), [])


class TestRechercheWeb(unittest.TestCase):
    """EF07-EF12. La logique est testée sans réseau ; un seul test sort."""

    def test_nettoyage_du_balisage_mediawiki(self):
        brut = 'Une <span class="searchmatch">eau</span> dite &laquo;&nbsp;potable&nbsp;&raquo;'
        propre = recherche_web._nettoyer(brut)
        self.assertNotIn("<", propre)
        self.assertNotIn("&laquo;", propre)
        self.assertIn("eau", propre)

    def test_pont_traduit_le_moore(self):
        moore, francais, explication = recherche_web.traduire_requete("koom")
        self.assertEqual(moore, "koom")
        self.assertEqual(francais, "eau")
        self.assertIn("corpus", explication)

    def test_pont_refuse_le_charabia(self):
        # Sans garde-fou, « zzzqxwv » était traduit en « presque totalité »
        # et le Web interrogé sur cette invention.
        moore, francais, _ = recherche_web.traduire_requete("zzzqxwv")
        self.assertEqual(francais, "zzzqxwv")

    def test_pont_depuis_le_francais(self):
        moore, francais, _ = recherche_web.traduire_requete("eau")
        self.assertEqual(francais, "eau")
        self.assertTrue(moore)

    def test_requete_vide(self):
        self.assertEqual(recherche_web.traduire_requete(""), ("", "", ""))

    def test_entrelacement_expose_le_moore(self):
        # Le modèle ne connaît pas le mooré et enterrait ces pages.
        moore = [{"id": f"m{i}"} for i in range(4)]
        autres = [{"id": f"a{i}"} for i in range(9)]
        melange = recherche_web._entrelacer(moore, autres)
        self.assertEqual(len(melange), 13)
        self.assertEqual(melange[0]["id"], "m0")
        dans_le_top6 = sum(1 for x in melange[:6] if x["id"].startswith("m"))
        self.assertGreaterEqual(dans_le_top6, 2)

    def test_entrelacement_sans_moore(self):
        autres = [{"id": f"a{i}"} for i in range(3)]
        self.assertEqual(recherche_web._entrelacer([], autres), autres)

    def test_recherche_reelle(self):
        """Seul test qui sort sur le réseau ; toléré s'il n'y a pas d'accès."""
        reponse = recherche_web.rechercher_web("koom", k=5)
        if not reponse["resultats"]:
            self.skipTest("pas d'accès réseau")
        self.assertIn("eau", reponse["terme_francais"])
        for resultat in reponse["resultats"]:
            for champ in ("titre", "extrait", "url", "source"):  # EF10, EF11
                self.assertIn(champ, resultat)
            self.assertTrue(resultat["url"].startswith("https://"))
            self.assertTrue(resultat["titre"])


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import api
        cls.client = api.app.test_client()

    def test_health(self):
        d = self.client.get("/api/health").get_json()
        self.assertEqual(d["status"], "ok")
        self.assertEqual(d["entrees"], 10566)
        self.assertIn("tfidf", d["backends"])

    def test_k_invalide_rend_400(self):
        for k in ("abc", "0", "-1", "", "1e9"):
            r = self.client.get(f"/api/search?q=eau&k={k}")
            self.assertEqual(r.status_code, 400, f"k={k!r}")
            self.assertIn("error", r.get_json())

    def test_k_borne(self):
        d = self.client.get("/api/search?q=eau&k=99999").get_json()
        self.assertLessEqual(len(d), 100)

    def test_domaine_inconnu_rend_400(self):
        self.assertEqual(self.client.get("/api/search?q=eau&domaine=Klingon").status_code, 400)

    def test_backend_inconnu_rend_400(self):
        self.assertEqual(self.client.get("/api/search?q=eau&backend=nawak").status_code, 400)

    def test_filtre_domaine_compose(self):
        # Une entrée « Agriculture, Water » doit sortir sous les deux puces.
        for dom in ("Agriculture", "Water"):
            d = self.client.get(f"/api/search?q=diguettes&k=20&domaine={dom}").get_json()
            self.assertTrue(all(dom in search.split_domaines(x["domaine"]) for x in d))

    def test_requete_vide(self):
        self.assertEqual(self.client.get("/api/search?q=").get_json(), [])

    def test_traduction(self):
        d = self.client.get("/api/translate?q=je+veux+boire+de+l%27eau").get_json()
        self.assertIn("glose", d)
        self.assertIn("idiomes", d)

    def test_phrase_trop_longue(self):
        r = self.client.get("/api/translate?q=" + "a" * 600)
        self.assertEqual(r.status_code, 400)

    def test_web(self):
        r = self.client.get("/api/web?q=koom&k=3")
        self.assertEqual(r.status_code, 200)
        self.assertIn("resultats", r.get_json())
        self.assertEqual(self.client.get("/api/web?q=eau&k=0").status_code, 400)
        self.assertEqual(self.client.get("/api/web?q=" + "a" * 300).status_code, 400)

    def test_liens(self):
        self.assertTrue(self.client.get("/api/liens?q=koom").get_json())
        self.assertEqual(self.client.get("/api/liens?q=").get_json(), [])

    def test_page_servie(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers.get("Content-Type", ""))

    def test_api_sans_index_semantique(self):
        """L'installation par defaut n'a pas l'index semantique.

        Ce cas n'etait couvert par aucun test, parce qu'ils s'executaient tous
        sur une machine ou l'index existe. L'API rejetait alors « auto » et
        « entrees » avec une erreur 409, ce qui rendait la recherche dans le
        corpus inutilisable sur toute machine fraichement installee.
        """
        vrai = embeddings.index_disponible
        embeddings.index_disponible = lambda: False
        try:
            for backend in ("auto", "entrees", "tfidf", ""):
                url = f"/api/search?q=koom&k=3&backend={backend}"
                r = self.client.get(url)
                self.assertEqual(r.status_code, 200, f"backend={backend!r}")
                self.assertTrue(r.get_json(), f"backend={backend!r}")
            # Ceux-la doivent bien etre refuses, avec un message explicite.
            for backend in ("semantique", "hybride"):
                r = self.client.get(f"/api/search?q=koom&backend={backend}")
                self.assertEqual(r.status_code, 409, f"backend={backend!r}")
            # Le glossage et la recherche Web ne dependent pas du modele.
            self.assertEqual(self.client.get("/api/translate?q=je+veux+manger").status_code, 200)
            self.assertEqual(self.client.get("/api/health").status_code, 200)
        finally:
            embeddings.index_disponible = vrai

    def test_cors_restreint(self):
        autorise = self.client.get("/api/search?q=eau&k=1", headers={"Origin": "null"})
        self.assertEqual(autorise.headers.get("Access-Control-Allow-Origin"), "null")
        bloque = self.client.get("/api/search?q=eau&k=1",
                                 headers={"Origin": "https://evil.example.com"})
        self.assertIsNone(bloque.headers.get("Access-Control-Allow-Origin"))


if __name__ == "__main__":
    if not SEMANTIQUE:
        print("Index sémantique absent : les tests qui en dépendent seront sautés.\n")
    unittest.main(verbosity=2)
