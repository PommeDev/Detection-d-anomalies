# Detection-d-anomalies

L’objectif principal de ce projet est de concevoir et d’entraîner un modèle de Deep Learning dédié à la détection d’anomalies dans un contexte industriel. Plus
précisément, nous travaillerons avec une base de données publique composée
d’images de câblages de voitures [(cf dataset)](#dataset). 

---------

Tout au long de ce projet, nous
seront amenés à explorer les étapes clés du développement et de l’application
d’un modèle d’apprentissage profond. 


Pour réaliser de la **détection d'anomalies** il existe plusieurs méthodes ([wikipédia](https://fr.wikipedia.org/wiki/D%C3%A9tection_d%27anomalies)), notamment la méthode des **k-plus proche voisins** ou bien les **modèles de markov cachés**. En réalité, il n'y a pas vraiment de prédominance d'une technique sur une autre et chaque méthode peut être performante si elle est utilisée dans des bonne situations avec les bon hyperparamétres [[1](#comparaison)].


Finalement, pour ce projet, l’approche retenue mettra l’accent sur l’utilisation d’**[autoencodeurs](#autoencodeurs)**, un type de réseau de neurones largement utilisé
pour les tâches de reconstruction et de détection d’anomalies. Bien que [[1](#comparaison)], parle du fait que Les autoencodeurs peuvent être efficaces pour détecter des anomalies dans des données complexes et de haute dimension. Cepedant, leur performance dépend fortement de la qualité des données d'entraînement et du réglage des hyperparamètres. Une limitation importante des autoencodeurs réside dans leur **sensibilité au sur-apprentissage**, **surtout si les données d'entraînement contiennent déjà des anomalies**.



## Analyse du sujet et déroulement


Nous allons utiliser des autoencodeurs afin de créer un modèle de Deep Learning permettant la détection d'anomalies industrielles d'une base de données d'images de cablage de voitures.
Pour ce qui est du regroupement des photos d'entraînement, nous avons utilisé le dataset [Engine Wiring](#dataset)


## Le dataSet
<a id="dataset"></a>

Nous utilisons le jeu de données de la base de données _AutoVi_ ([lien](#link_dataset)), un dataset proposé par Renault Group, qui recense de nombreuses images. Pour ce projet nous allons nous consentrer exclusivement sur les défaillances dans le cablage. Ainsi nous regarderons uniquement **Engine Wiring**.

Voici un exemple d'une image du jeu de données 

<img src="0000.png" alt="Image du jeu d'entrainement" width="100" height="100">

Se sont des images au format : **400x400 pixels**


## Utilisation de modèle autoencodeurs
<a id="autoencodeurs"></a>


<img src="Figure_1.png" alt="Image d'autoencodeur" width="200" height="200">

----

**Apprentissage non supervisé (images pas étiquetées)** [[2](#ibm1)]
*But*: compresser (**encoder**) et décompresser (**décoder**) les données. <br>
Il met en valeur les variables latentes (ce qui fait varier la catégorisation des images), pour les choisir: **goulot d'étranglement**<br>  
**Espace latent**: ensemble des variables latentes.<br>
l'autoencodeurs choisira ensuite quelles _variables latentes_ il garde pour décoder au mieux (le plus identiquement possible à l'original) les images
taches génératives d'immages: (**VAE et AAE**)<br>
Ces frameworks interviennent dans divers modèles d’apprentissage profond : par exemple dans les architectures de réseaux neuronaux convolutifs (CNN), utilisées dans les tâches de vision par ordinateur comme la segmentation d’images, ou dans les architectures de réseaux neuronaux récurrents (RNN), utilisées dans les tâches de séquence à séquence (seq2seq).

**Segmentation d'image**: segmenter l'image en groupes de pixels ayant les mêmes caractéristiques appelés masques de segmentation/ou classes sémantiques (et ça c'est de l'apprentissage supervisé?!).<br>
**apprentissage autosupervisé** des autoencodeurs: car ce n'est pas supervisé mais la sortie est comparée à l'entrée donc on ne peut pas dire qu'il n'y a aucune supervisation <br>
encodeurs: _réduit la dimension_ (de moins en moins de neurones par couches)
goulot d'étranglement (le moins de noeuds; couche d'entrée du décodeur et couche de sortie de l'encodeur) (le code en sortant c'est la représentation de l'espace latent)
décodeurs: décompressement et comparaison à la vérité terrain (entrée d'origine), il y aura des erreurs de reconstruction. Il peut être supprimé à la fin, dans les applications de génération d'image (par exemple). <br>
Choix d'un autoencodeur: le type de réseau neuronal; la taille du code, nombre de couches, nombre de noeud par couches et la fonction de perte.
Structures d'autoencodeurs: 
* **sous-complets**: taille goulot d'étranglement fixe
* **régularisés**: modification du calcul de l'erreur de reconstruction:
  - épars (SAE): réduction du nb de noeuds ACTIVES à l'aide d'une fonction de parcimonie
  - contractifs: terme de régularisation qui pénalise le réseau lorsqu’il modifie la sortie, en réponse à des changements insuffisants dans l’entrée (enlève les bruits)
  - débruiteurs (DAE): même style mais pas de vérité terrain
* **Variationnels** (VAE): GENERATION crée des nouveaux échantillons de données en variant des paramètres (repose sur la distribution de probas) !
  



 ### Erreur de reconstruction 

 Plusieurs formules existe : 
  **Erreur Moyenne Quadratique** (MSE) : $\frac{1}{n}\sum (y_i - e_i)^2, \space e_i \text{ vrai valeur}, y_i \text{  valeur prédite}$

  **Erreur Moyenne Absolue** (MAE) : $\frac{1}{n}\sum |y_i-e_i| , \space e_i \text{ vrai valeur}, y_i \text{  valeur prédite}$ 

  **Entropie Croisée** : $- \sum_x p(x)\log q(x) , \space p \text{ vraie distribution }, q \text{ distribution observée}$


## Sources
  
<a id="comparaison"></a>
[1]Campos, G.O., Zimek, A., Sander, J. et al. On the evaluation of unsupervised outlier detection: measures, datasets, and an empirical study. Data Min Knowl Disc 30, 891–927 (2016). https://doi.org/10.1007/s10618-015-0444-8


<a id="ibm1"></a>
[2] IBM : https://www.ibm.com/fr-fr/think/topics/autoencoder

<a id="link_dataset"></a>
https://autovi.utc.fr/index_fr.html

<p>
License:
Copyright © 2023-2024 Renault Group
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of the license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/.
</p>