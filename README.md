link al file colab con i comandi e con le celle per fine-tuning e valutazione di segformer: https://colab.research.google.com/drive/1_q9pZcAzU3vpXVue3c7ehwbQIbVJ2MqW?usp=sharing
TRAINING
il training viene effettuato sul dataset contenuto in train_roweeder_effective
le configurazioni, oltre che da linea di comando sono scritte nel file UKAN/Seg_UKAN/config.yml.
Per la loss, che risulta essere da file BCDiceLoss, è in realtà CrossEntropyLoss come hardcodato nello script di training UKAN/Seg_UKAN/train.py

TESTING
il testing si effettua su test_roweeder_complete.
Le configurazioni sono contenute, oltre che da liena di comando, anche nel file trained_models/UKAN/config.yml.
cambiare nel file:
- data_dir inserendo la directory dove si trova test_roweeder_complete
- input_list inserendo gli iperparametri del modello che si intende utilizzare
