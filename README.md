# DESCRIZIONE DEL PROGETTO
Si è testato il modello UKAN messo a disposione al seguente link [UKAN for medical purpose](https://github.com/CUHK-AIM-Group/U-KAN.git), e confrontato con la versione più leggera di Segofrmer messa a disposizione dalla libreria transformes di HUgging Face. 
Tra i principali cambiamenti operati sulla repo originale di UKAN c'è la sua trasformazione da modello mutlilabel a classi esclusive: in particolare il cambiamento della loss in CrossEntroyLoss.
Altri cambiamenti sono strettamente relativi al funzionamento esecutivo del modello, o alla stampa delle immagini finali della segmentazione, o all'aggiunta di metriche e script per la valutazione. 

Sia prima di lanciare Segfomer che UKAN nel notebook [DL-Project](https://colab.research.google.com/drive/1_q9pZcAzU3vpXVue3c7ehwbQIbVJ2MqW?usp=sharing) eseguire le prime celle di installazione
# UKAN
per compatibilità di path inserire Seg_UKAN in una cartella UKAN.
## TRAINING
### Dataset
il training viene effettuato sul dataset contenuto in train_roweeder_effective.<br>
Sono le immagini 512 x 512 pixels, dei campi 000, 001, 002, 004 non contenenti solo background.
### Configurazioni
la configurazioni passate da linea di comando come parametri hanno la "precedenza". <br>
Oltre che da linea di comando delle configurazioni sono scritte nel file [Seg_UKAN/config.py]().

Per la loss, CrossEntropyLoss, è sia passata come parametro da linea di comando che hardcodata nello script di training [Seg_UKAN/train.py](). <br>
Gli iperparametri per la complessità del modello sono indicati tramite il parametro --input_list.
### Outputs
Vengono forniti i pesi del modello in un file model.pth, e le configurazioni di trainig in un file config.yml.<br>
Questi file vengono creati in una directory il cui nome è composto dai parametri di training da linea di comando [--output_dir ('outputs' di default)/--name];
### Notebook
Il comando per l'addestramento è presente nella sezione "UKAN/Roweeder Dataset/Trainig" del seguente fle colab: [DL-Project](https://colab.research.google.com/drive/1_q9pZcAzU3vpXVue3c7ehwbQIbVJ2MqW?usp=sharing).<br>
In realtà l'addestramento è stato fatto su kaggle, per questioni di disponibilità di tempo-GPU, e le differenze dei parametri da linea di comando rispetto a colab sono:
- le epoche: 400
- input_list: siaono provate le tre configurazioni [128, 160, 256], [64, 80, 128], [32, 40, 64]
- data_dir: mettere il path di train_roweeder_effective
### File ottenuti
in [trained_models/UKAN]() sono contenuti i:
- model.pth ottenuti per le tre configurazioni
- config.yml ottenuto per [64, 80, 128]
## TESTING
### Dataset
il training viene effettuato sul dataset contenuto in test_roweeder_complete.<br>
Sono le immagini 512 x 512 pixels, del campo 003.
### Configurazioni
la configurazioni passate da linea di comando come parametri hanno la "precedenza". <br>
Oltre che da linea di comando delle configurazioni sono scritte nel file [trained_models/UKAN/config.yml](). In questo file bisogna cambiare:
- data_dir: mettere il path di test_roweeder_complete
- input_list: mettere le configurazioni di iperparametri corrispondenti al modello testato
I pesi dei modelli sono presenti nei file [trained_models/UKAN/model.pth]()

Per compatibilità di path è importante dire che gli script di valutazione vanno a prendere:
-  il file config.yml e model.pth sono cercati nella directory il cui nome è composto dai parametri di valutazione da linea di comando [--output_dir ('outputs' di default)/--name];
-  rinominare i file model-64.pth o model-32.pth, come "model.pth", nel caso si voglia testare uno di questi modelli
### Script di valutazione
Sono presenti tre script di valutazione:
- [Seg_UKAN/val-multiclass.py](): calcola la F1-score per ognuna delle classi, mette nella cartella [--output_dir ('outputs' di default)/--name/output_vals] le maschere predette dal modello per ogni classe;
- [Seg_UKAN/val-complexity.py](): calcola il numero di parametri e GMACs
- [Seg_UKAN/val-inference_time2.py](): calcola il tempo di inferenza usando la classe Event di torch se si ha a disposizione la GPU, la libreria time (più imprecisa) altrimenti
### Notebook
I comandi per le valutazioni sono presenti nella sezione "UKAN/Roweeder Dataset/Evaluation" del seguente fle colab: [DL-Project](https://colab.research.google.com/drive/1_q9pZcAzU3vpXVue3c7ehwbQIbVJ2MqW?usp=sharing).<br>

# Segformer MiT-b0
Sia prima del fine tuning che della Evaluation eseguire le celle per l'import che sono presenti nella sezione "Segformer" del fle colab: [DL-Project](https://colab.research.google.com/drive/1_q9pZcAzU3vpXVue3c7ehwbQIbVJ2MqW?usp=sharing)
## FINE TUNING
### Dataset
il training viene effettuato sul dataset contenuto in train_roweeder_effective.<br>
Sono le immagini 512 x 512 pixels, dei campi 000, 001, 002, 004 non contenenti solo background.
### Configurazioni
la configurazioni che in UKAN venivano passate da linea di comando, come la directory dei dati di training, sono passate nelle costanti ad inizio della cella.<br>
### Outputs
Vengono forniti i pesi del modello in un file model.pth<br>
Questi file vengono creati in una directory il cui nome è composto dai parametri di training [OUTPUT_DIR/EXP_NAME];
### Notebook
la cella per l'addestramento è presente nella sezione "Segformer/Fine Tunining" del seguente fle colab: [DL-Project](https://colab.research.google.com/drive/1_q9pZcAzU3vpXVue3c7ehwbQIbVJ2MqW?usp=sharing).<br>
In realtà l'addestramento è stato fatto su kaggle, per questioni di disponibilità di tempo-GPU, e la differenza dei parametri di addestramento rispetto a colab è:
- data_dir: mettere il path di train_roweeder_effective
### File ottenuti
in [trained_models/Segformer]() è contenuto il:
- model.pth ottenuto dopo il fine-tuning
## TESTING
### Dataset
il training viene effettuato sul dataset contenuto in test_roweeder_complete.<br>
Sono le immagini 512 x 512 pixels, del campo 003.
### Configurazioni
la configurazioni che in UKAN venivano passate da linea di comando, come la directory dei dati di training, sono passate nelle costanti ad inizio della cella.<br>
Assicurarsi che il parametro:
- DATA_DIR: contiene il path di test_roweeder_complete
I pesi dei modelli sono presenti nei file [trained_models/segformer/model.pth]()

Per compatibilità di path è importante dire che le celle di valutazione vanno a prendere:
-  il file model.pth nella directory il cui nome è composto dai parametri di valutazione OUTPUT_DIR/EXP_NAME];
### Script di valutazione
Sono presenti due celle di valutazione:
1. calcola la F1-score per ognuna delle classi, mette nella cartella [OUTPUT_DIR/EXP_NAME/output_vals] le maschere predette dal modello per ogni classe;
2. calcola il tempo di inferenza usando la classe Event di torch se si ha a disposizione la GPU, la libreria time (più imprecisa) altrimenti
### Notebook
le celle per le valutazioni sono presenti nella sezione "Segformer/Valutazione" del seguente fle colab: [DL-Project](https://colab.research.google.com/drive/1_q9pZcAzU3vpXVue3c7ehwbQIbVJ2MqW?usp=sharing).<br>
