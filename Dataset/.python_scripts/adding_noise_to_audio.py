import librosa
import numpy as np
import soundfile as sf
import os

def add_noise(input_folder, output_folder):
    try:
        # Créer le dossier de sortie s'il n'existe pas
        os.makedirs(output_folder, exist_ok=True)

        # Parcourir tous les fichiers audio du dossier
        for fichier_audio in os.listdir(input_folder):
            if fichier_audio.endswith(".wav"):
                input_path = os.path.join(input_folder, fichier_audio)
                output_path = os.path.join(output_folder, f"noise_added_{fichier_audio}")

                # Charger l'audio
                y, sr = librosa.load(input_path, sr=22050)

                # Ajouter du bruit
                bruit = np.random.randn(len(y)) * 0.0002 # very low noise due low heart pulse frenquecy
                y_bruit = np.clip(y + bruit, -1.0, 1.0)

                # Sauvegarde du fichier
                sf.write(output_path, y_bruit.astype(np.float32), sr)

        print("✅ Augmentation terminée avec succès !")

    except Exception as e:
        print(f"❌ Erreur : {e}")

# Exemple d'utilisation
add_noise("../Dataset6/multi_disease", "../Dataset6/multi_disease_added_noise")