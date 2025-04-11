e# UKAN
per compatibilità di path inserire Seg_UKAN in una cartella UKAN.
## TRAINING
### Dataset
il training viene effettuato sul dataset contenuto in train_roweeder_effective.<br>
Sono le immagini dei campi 000, 001, 002, 004 non contenenti solo background.
### Configurazioni
la configurazioni passate da linea di comando come parametri hanno la "precedenza". <br>
Oltre che da linea di comando delle configurazioni sono scritte nel file [Seg_UKAN/config.py]().

Per la loss, CrossEntropyLoss, è sia passata come parametro da linea di comando che hardcodata nello script di training [Seg_UKAN/train.py](). <br>
Gli iperparametri per la complessità del modello sono indicati tramite il parametro --input_list.
### Outputs
Vengono forniti i pesi del modello in un file model.pth, e le configurazioni di trainig in un file config.yml.<br>
Questi file vengono creati in una directory il cui nome è composto dai parametri di training [--output_dir ('outputs' di default)/--name];
### Notebook
Il comando per l'addestramento è presente nella senzione "UKAN/Roweeder Dataset/Trainig" del seguente fle colab: [DL-Project](https://colab.research.google.com/drive/1_q9pZcAzU3vpXVue3c7ehwbQIbVJ2MqW?usp=sharing).<br>
In realtà l'addestramento è stato fatto su kaggle, per questioni di disponibilità di tempo-GPU, e le differenze dei parametri da linea di comando rispetto a colab sono:
- le epoche: 400
- input_list: siaono provate le tre configurazioni [128, 160, 256], [64, 80, 128], [32, 40, 64]
- data_dir: mettere il path di train_roweeder_effective
### File ottenuti
in [trained_models]() sono contenuti i:
- model.pth ottenuti per le tre configurazioni
- config.yml ottenuto per [64, 80, 128]
## TESTING

il testing si effettua su test_roweeder_complete.
Le configurazioni sono contenute, oltre che da liena di comando, anche nel file trained_models/UKAN/config.yml.
cambiare nel file:
- data_dir inserendo la directory dove si trova test_roweeder_complete
- input_list inserendo gli iperparametri del modello che si intende utilizzare
