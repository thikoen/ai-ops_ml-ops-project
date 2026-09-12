# ICE Hamburg–Berlin: ein minimales MLOps-Projekt

CAS AI Operations, **Feature–Training–Inference (FTI)**: Wird ein ICE von Hamburg Hbf
nach Berlin Hauptbahnhof mindestens fünf Minuten verspätet ankommen? Die Anfrage
wird beim beobachteten Losfahren in Hamburg simuliert.

Drei Python-Pipelines verwenden **Hopsworks** als Featurestore und Model Registry.
**MinIO** speichert Rohdaten, Quellenmetadaten, Berichte und Vorhersagen. Docker
Compose enthält nur MinIO und einen wiederverwendbaren Python-Pipeline-Container.

## Unsere Entscheidungen – einschliesslich der verworfenen Wetterhypothese

**Ursprüngliche Strecke:** S3 Fehraltorf → Zürich HB. In der ersten Validierung
waren von 75 Fahrten nur zwei verspätet; auch der Trainingsanteil enthielt nur zwei
positive Beispiele. Das Modell sagte stets „pünktlich“. Accuracy 97,3 % bei F1 = 0
war hier kein Erfolg. Wir entschieden deshalb, eine andere Strecke und mehr Historie
zu verwenden, statt die schwachen Ergebnisse allein zu dokumentieren.

**Wechsel nach Deutschland:** Eine Vorprüfung der August-Daten 2025 ergab für ICE
Hamburg Hbf → Berlin Hauptbahnhof 585 passende Fahrten mit vollständigen Zeitangaben,
davon 305 mit mindestens fünf Minuten Ankunftsverspätung. Beide Klassen waren damit
wesentlich besser vertreten. Die finale Verarbeitung berücksichtigt zusätzlich
Monatsgrenzen und den lokalen Abfahrtstag; ihre Monatszahlen weichen deshalb leicht
von der Vorprüfung ab. Die konkrete Verbindung wurde anhand der Datenmenge und
Klassenverteilung ausgewählt, nicht anhand des späteren Oktober-Testergebnisses.

**Wetter entfernt:** Die Annahme „Temperatur hilft bei der Vorhersage“ wurde geprüft.
Auf derselben chronologischen August-Validierung ergaben sich:

| Merkmale | F1 |
|---|---:|
| Historische Durchschnittsverspätung + aktuelle Abfahrtsverspätung + Temperatur | 0,470 |
| Historische Durchschnittsverspätung + aktuelle Abfahrtsverspätung | 0,470 |
| Dieselben zwei Merkmale + Uhrzeit + Wochenende | 0,498 |

Mit und ohne Temperatur waren die Klassen-Vorhersagen identisch. Der Wetterabruf
wurde daher vollständig entfernt. Der kleine Vorteil der Kalendermerkmale ist ein
Ergebnis dieser Validierung, kein statistisch gesicherter allgemeiner Effekt.
**Wir behaupten nicht, dass Wetter generell keine Verspätungen verursacht.** Wir
haben lediglich in diesem Modell und Zeitraum keinen zusätzlichen Nutzen nachgewiesen.

**Modell:** Logistische Regression, weil sie klein und nachvollziehbar ist. Ein
begrenzter Vergleich auf der früheren Variante mit drei Features ergab für
`GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=42)` F1 0,360
gegenüber 0,470 für logistische Regression. Wir haben deshalb keinen komplizierteren
Modelltyp eingeführt. Die finale Variante verwendet vier Features und keinen Wetterdienst.

Datenquelle, Zeitraum und Merkmale wurden gemeinsam verändert. Der Fortschritt
gegenüber der S3 ist daher nicht allein dem Länderwechsel zuzuschreiben.

## Architektur und Features

```text
Archivierte DB-Daten (Parquet)
            │
    Feature-Pipeline ───────── Rohdaten/Quellenmetadaten → MinIO
            │
 Hopsworks Feature Group
            │
  Hopsworks Feature View
            │
    Training-Pipeline ──────── Evaluationsbericht → MinIO
            │
 Hopsworks Model Registry
            │
Feature View + simulierte aktuelle Abfahrtsbeobachtung
            │
    Inferenz-Pipeline ──────── Vorhersagen als Parquet → MinIO
```

| Feature | Art | Bedeutung |
|---|---|---|
| `delay_mean_7d` | Aggregiert | Mittlere Ankunftsverspätung der Verbindung in sieben abgeschlossenen Tagen. Für Prognosetag D: D−8 bis D−2. |
| `departure_delay_minutes` | Aktuell, im Replay simuliert | Archivierte Abfahrtszeit in Hamburg minus geplante Abfahrtszeit. |
| `departure_hour` | Bei Anfrage bekannt | Lokale Stunde inklusive Minutenanteil, z. B. 10,5 für 10:30 Uhr. |
| `is_weekend` | Bei Anfrage bekannt | 1 am Samstag/Sonntag, sonst 0. |

Die Abfahrtsverspätung erfüllt die RT-Feature-Anforderung. Die Prognose erfolgt
**bei beobachteter Abfahrt**, nicht bereits vor der geplanten Abfahrt. Andernfalls
wäre diese Information noch nicht bekannt und ihre Verwendung wäre Datenleckage.

## Starten in Codespaces oder mit lokalem Docker

Voraussetzungen: eine laufende Linux-Container-Engine, Docker Compose v2, Internet
und ein Hopsworks-Projekt. `docker info` muss auch einen Server anzeigen. Python
**3.12** und alle Pakete aus `uv.lock` werden im Container installiert.

In GitHub **Code → Codespaces → Create codespace on main** wählen. Im Terminal des
Repositorys ausführen; die gleichen Befehle gelten unter lokalem Linux:

```sh
# Eine bestehende .env nicht überschreiben.
cp .env.example .env
```

In `.env` die Beispielwerte durch eigene Werte ersetzen:

- `HOPSWORKS_HOST`: Host aus der Projekt-URL, ohne https:// und ohne Pfad.
- `HOPSWORKS_PROJECT`: eigener Projektname.
- `HOPSWORKS_API_KEY`: eigener Key mit Lese-/Schreibzugriff auf Features und Modelle.
- `MINIO_ROOT_USER` und `MINIO_ROOT_PASSWORD`: eigene lokale Zugangsdaten;
  das Passwort muss mindestens acht Zeichen haben.

Die lokalen Zugangsdaten werden **nicht** in Codespaces übertragen. `.env` und
SDK-Zertifikate sind von Git und dem Image ausgeschlossen. Unter PowerShell lautet
der Kopierbefehl `Copy-Item .env.example .env`.

```sh
docker compose up -d minio
docker compose build pipeline

# 1. Features und Labels in Hopsworks schreiben.
docker compose run --rm pipeline features --start 2025-06-01 --end 2025-10-30

# 2. Nur Vergangenheit vor dem September-Cutoff zum Training verwenden.
docker compose run --rm pipeline train --cutoff 2025-09-01

# 3. Einen zurückgehaltenen Tag als aktuelle Anfragen simulieren.
docker compose run --rm pipeline infer --day 2025-10-30
```

Die Feature-Pipeline lädt automatisch acht frühere Tage als Vorlauf und einen Tag
nach `--end` für Ankünfte über Mitternacht. Da die Quelle Monatsdateien bereitstellt,
werden hier Mai–Oktober eingelesen: zusammen rund **624 MB**. Das Ende am 30. Oktober
hält auch den Folgetag im selben Monatsarchiv. Andere Monatsgrenzen können eine
zusätzliche Datei erfordern.

Jede Datei wird heruntergeladen, gegen ihren veröffentlichten SHA-256 geprüft und
in Blöcken gefiltert. Nur ICE-Halte an den beiden Stationen bleiben im MinIO-Cache.
Unterbrochene Downloads werden bis zu zweimal wiederholt; unvollständige Dateien
werden nicht als gültige Extrakte gespeichert. Folgeläufe verwenden den Cache.
Akzeptiert werden Monate, die in der festgelegten Dataset-Revision existieren.
Für den Inferenztag müssen bereits Features im Featurestore vorhanden sein.

### Ergebnisse ansehen

- **Hopsworks:** `ice_hamburg_berlin_examples` (Feature Group),
  `ice_hamburg_berlin_prediction` (Feature View), `ice_hamburg_berlin_classifier` (Modell).
- **Terminal:** Datenzahlen, Validierungsmetriken und die ersten Vorhersagen.
- **MinIO:** Bucket `rail-delay`, darin `raw/`, `reports/`, `models/latest.json`
  und `predictions/<datum>/<lauf-id>.parquet`.
- **MinIO-Konsole:** lokal [localhost:9001](http://localhost:9001). In Codespaces
  Port `9001` im Reiter **Ports** privat weiterleiten und dessen URL öffnen.

`docker compose down` stoppt die Services und erhält das benannte MinIO-Volume.
Ein Training erstellt eine neue Modellversion. Erst nach erfolgreichem Speichern
von Modell und Bericht wird das Manifest `models/latest.json` aktualisiert.

## Daten und Label

Quelle: [Deutsche Bahn Data von Piet Brömmel](https://github.com/piebro/deutsche-bahn-data),
veröffentlicht auf [Hugging Face](https://huggingface.co/datasets/piebro/deutsche-bahn-data).
Grundlage sind öffentliche DB-APIs. **Attribution: Deutsche Bahn und Piet Brömmel;
Datenlizenz CC BY 4.0.** Dieses Projekt filtert und transformiert die Daten.
Die Dataset-Revision `6edf2ef689245f4e291cf0599149519d899eeae5` ist fixiert. Abrufzeit,
Original-URL, Revision, Grösse und Dateihash werden in MinIO dokumentiert.

Die Monatsdateien enthalten geplante und archivierte geänderte Zeiten. Dieselbe
Fahrt wird an beiden Stationen verbunden. Der Schlüssel enthält auch das Datum der
Fahrt, nicht nur die wiederkehrende Fahrtkennung. Gegenrichtung, ausgefallene relevante
Halte, unklare doppelte Halte, fehlende Zeiten und schon vor der Anfrage erfolgte
Ankünfte werden ausgeschlossen. Geplante Fahrzeit: plausibler Bereich 1–360 Minuten.

**Label `delayed_5min`:** `(arrival_change_time - arrival_planned_time) >= 5 Minuten`
in Berlin. Dies ist unsere Projektdefinition, nicht die offizielle DB-Pünktlichkeitsschwelle.
Ausfälle gelten nicht als pünktliche Fahrten. Fehlende Zeiten werden nicht durch
Null-Verspätungen ersetzt. Die Quelle verarbeitet und ergänzt selbst API-Felder;
wir übernehmen ihre archivierte Sicht, nicht unabhängige physische Messungen.

`trip_id` ist der Primary Key, `event_time` die beobachtete Abfahrt in UTC. Beide
sind keine Modell-Features. Ortszeiten werden mit `Europe/Berlin` in UTC umgerechnet;
unklare Zeiten beim Sommerzeitwechsel werden verworfen. `label_available_at` dient
als Trainings-Hilfsspalte und geht ebenfalls nicht in das Modell ein.

## Die drei Pipelines und ihre zeitliche Trennung

**Feature:** Quelle laden → Rohdaten sichern → Fahrten verbinden und Label berechnen
→ Vergangenheit aggregieren und aktuelle Merkmale ableiten → Dataframe in Hopsworks
Feature Group schreiben. Der Upload wird abgewartet.

**Training:** Feature Group laden → Feature View mit Features und Label erstellen
→ versioniertes Training Dataset lesen → chronologisch trainieren/validieren → Modell
mit `joblib` speichern und in Hopsworks registrieren. Standardisierung und logistische
Regression sind dieselbe gespeicherte Scikit-Learn-Pipeline. `class_weight="balanced"`
berücksichtigt die Klassenhäufigkeiten; die Klassengrenze bleibt bei Wahrscheinlichkeit 0,5.

Die ersten ungefähr 80 % der erlaubten Trainingstage trainieren das Validierungsmodell,
spätere Tage validieren es. Nicht rechtzeitig verfügbare Labels werden aus dem
Trainingsanteil entfernt. Nach der Evaluation wird das gespeicherte Modell auf allen
erlaubten Trainingstagen neu trainiert. Berichtsmetriken beziehen sich auf die
Validierung vor diesem abschliessenden Refit.

**Inferenz:** Konkrete Modellversion aus dem Manifest laden → zugehörige Feature View
initialisieren → historische Features für den simulierten Tag lesen → aktuelle
Abfahrtsverspätung aus Hamburger Quelldaten neu einlesen → Kalendermerkmale identisch
berechnen → vorhersagen → Ergebnis in MinIO speichern. **Das Berliner Ankunftslabel
wird dabei nicht verwendet.** Die Feature View liefert keine Labels zur Inferenz.

Für historische Labels gilt im Replay die Annahme: verfügbar ab dem zweiten lokalen
Kalendertag nach Ankunft. Deshalb endet das Siebentagefenster bei D−2. Bei Cutoff
1. September werden Abfahrten vor dem 31. August verwendet, zusätzlich mit
`label_available_at <= Cutoff`. Der 31. August ist Publikationspuffer; Inferenz ist
frühestens ab 1. September zulässig. Diese Prüfung verhindert Überschneidungen.

## Ergebnisse und Grenzen

Auf derselben August-Validierung wurden die Features gewählt. September war bereits
während der Entwicklung betrachtet worden. Für die finale Variante wurde deshalb
zusätzlich **1.–30. Oktober** geprüft, ohne daran Modell oder Schwelle anzupassen.

- Feature-Zeilen Juni–30. Oktober: **3057**.
- Trainingsanteil vor Validierung: **1591**, davon **718** verspätet.
- Validierung ab 12. August: **357**, davon **201** verspätet.
- Abschliessendes Refit vor September-Cutoff: **1967** Fahrten.
- Oktober-Test: **516**, davon **336** verspätet.

| Oktober-Test | Precision | Recall | F1 | Accuracy | Balanced Accuracy |
|---|---:|---:|---:|---:|---:|
| Logistische Regression | 0,786 | 0,405 | 0,534 | 0,541 | 0,600 |
| Abfahrtsregel: Abfahrt >= 5 Min. → Ankunft verspätet | 0,837 | 0,321 | 0,465 | 0,517 | 0,602 |
| Immer verspätet | 0,651 | 1,000 | 0,789 | 0,651 | 0,500 |
| Mehrheitsklasse aus Training: immer pünktlich | 0,000 | 0,000 | 0,000 | 0,349 | 0,500 |

Das Modell erkennt **136 von 336** verspäteten Fahrten und erzeugt **37 Fehlalarme**.
Es erkennt mehr Verspätungen als die Abfahrtsregel, erzeugt aber auch mehr Fehlalarme.
Balanced Accuracy ist praktisch gleich. „Immer verspätet“ erreicht wegen der häufigen
Verspätungen sogar höheren F1 und höhere Accuracy, warnt jedoch bei allen 180 pünktlichen
Fahrten falsch. **Ein hoher F1 allein beweist daher keinen praktischen Mehrwert.**
Das Modell hat höhere Warnpräzision, verpasst weiterhin viele Verspätungen und ist
kein verlässlicher Reisedienst. Seine Leistung wird nicht als generell überlegen dargestellt.

Weitere Einschränkungen:

- **Historischer Replay statt Live-Service:** Die Aufgabe erlaubt simulierte aktuelle
  Anfragen. Monatsdaten sind rückblickend verarbeitet. Die Abfahrtszeit wird im Replay
  so behandelt, als wäre sie bei Abfahrt bekannt; ein historisch garantierter
  As-of-Datenstand lässt sich damit nicht nachweisen.
- Die angenommene Label-Verfügbarkeit ist keine Publikationsgarantie des Monatsarchivs.
- Der Anbieter dokumentiert Erfassungslücken. Fehlende Fahrten und ausgeschlossene
  Ausfälle können die Verteilung verzerren, auch bei vollständig geladenen Dateien.
- Baustellen, Netzbelegung und weitere Einflüsse fehlen. Die einfache lineare
  Modellierung der Uhrzeit ist eine weitere Vereinfachung.
- Bei geänderter Feature-Semantik `HOPSWORKS_FEATURE_VERSION` erhöhen und alle drei
  Schritte erneut ausführen. Bestehende Views werden nicht stillschweigend umdefiniert.

## Technischer Prüfstand und Tests

Stand **12.09.2026**: 15 automatisierte Tests bestanden; Ruff ohne Befund.
Echte Daten-/Modelltests mit den oben angegebenen Zeiträumen wurden ausgeführt.
Compose-Konfiguration und Linux-Abhängigkeitsauflösung aus `uv.lock` werden geprüft.
Echter Hopsworks-SDK-Login und Metadatenzugriff auf Featurestore und Registry waren
möglich (Backend 5.0.3, SDK 5.0.7).

**Offen für Codespaces:** Container-Build und vollständiger Lauf MinIO → Hopsworks-
Upload → Training Dataset → registriertes Modell → Inferenz. Lokal lief keine
Docker-Engine. Der native Windows-Versuch endete vor dem Upload an der nicht für
Windows bereitgestellten Hopsworks-Delta-Bibliothek. Im Linux-Container wird sie
über `hopsworks[python]` installiert. Lokale Dateicache- und S3-Mock-Tests ersetzen
diesen Integrationstest nicht. Im Hopsworks-Projekt wurden noch keine Features
oder Modelle durch diese Implementierung angelegt.

Unter Linux mit Python 3.12 und uv:

```sh
uv sync --locked
uv run pytest -q
uv run ruff check src tests
```

Der Dockerfile installiert `build-essential`, weil die SDK-Abhängigkeit `twofish`
einen C-Compiler benötigt. Versionen sind über `pyproject.toml` und `uv.lock` definiert.
Die Tests prüfen Schlüssel, Fahrtrichtung, Labelqualität, Zeitfenster, aktuelle Inputs,
Trainingssplit, Replay-Cutoff, Download-Hashes und S3-Parquet/JSON-Roundtrips.

| Anforderung | Umsetzung |
|---|---|
| Drei FTI-Pipelines | Drei Funktionen in `src/rail_delay/pipelines.py` |
| Aggregiertes Feature | Siebentägige Durchschnittsverspätung |
| Aktuelles Feature | Simulierte aktuelle Abfahrtsverspätung |
| Hopsworks FG / PK / Event Time | `ice_hamburg_berlin_examples` / `trip_id` / `event_time` |
| Feature View / Training Dataset | Versionierte Metadaten in der Trainings-Pipeline |
| Modell trainieren, evaluieren, speichern, laden | Scikit-Learn und Hopsworks Model Registry |
| Object Store als Projektwunsch | MinIO |
| Dokumentation und Einrichtung | README, Dockerfile, Compose, `.env.example`, `uv.lock` |
