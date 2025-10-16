# Aktienkurs-Vorhersage

Dieses Projekt  verwendet historische Aktienkurse, um zukünftige Schlusskurse vorherzusagen. Es implementiert verschiedene Modelle, darunter lineare Regression, Ridge, Lasso, polynomiale Regression und ein neuronales Netzwerk.


## Features

- Historische Preise (Close)
- Gleitende Durchschnitte (MA5, MA10)
- Tagesrenditen
- Volatilität (Standardabweichung der Renditen)
- Vorhersage des nächsten Tagespreises

## Modelle

1.  **Lineare Regression**
2.  **Ridge Regression**
3.  **Lasso Regression**
4.  **Polynomial Regression (Grad 2)**
5.  **Neuronales Netzwerk** (Feedforward, 2 versteckte Schichten)

## Installation

1.  Python 3.x installieren (empfohlen: >=3.10)
2.  Optional: Virtuelle Umgebung erstellen:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # macOS/Linux
    venv\Scripts\activate     # Windows
    ```

3.  Benötigte Pakete installieren:

    ```bash
    python3 -m pip install scikit-learn yfinance pandas numpy matplotlib tensorflow
    ```

## Nutzung

1.  Skript ausführen:

    ```bash
    python3 aktienanalyse.py
    ```

2.  Das Skript gibt aus:
    * MSE, MAE, R² für jedes Modell
    * Diagramme der Vorhersagen vs. tatsächliche Schlusskurse
    * Empfehlung für den nächsten Tag ("Kaufen" oder "Verkaufen")

## Hinweise

* Die Vorhersagen basieren auf historischen Daten und dienen **nicht als Finanzberatung**.
* Die Genauigkeit hängt stark von den verwendeten Features und dem Modell ab.
* Für das neuronale Netz ist TensorFlow erforderlich; GPU-Beschleunigung kann optional genutzt werden.

## Beispielausgabe
