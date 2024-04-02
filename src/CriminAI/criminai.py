import tkinter as tk
from PIL import ImageTk, Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.AlgoGenetique import algo_genetique
from utils.VAE import autoencodeur_celebA as autoencodeur
from utils.VAE.autoencodeur_celebA import CustomDataset
import os
import numpy as np

# Définir les dimensions
latent_dim = 64

# Charger le modèle sauvegardé
checkpoint = torch.load("src/CriminAI/models/best_model.pth")
autoencoder = autoencodeur.VAE(latent_dim)
autoencoder.load_state_dict(checkpoint)
autoencoder.eval()  # Mettre le modèle en mode évaluation

celeba_data_dir = "src/CriminAI/image_batch" # Chemin vers le dossier contenant les images = Chemin où les données CelebA sont extraites
# Transformation des images
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Redimensionner les images à une taille de 64x64 pixels
    transforms.ToTensor(),  # Convertir les images en tenseurs PyTorch
])
# Charger les données à partir du dossier img_align_celeba
celeba_dataset = CustomDataset(root_dir=celeba_data_dir, transform=transform)
# Définir un DataLoader pour la gestion des données
batch_size = 64
data_loader = DataLoader(celeba_dataset, batch_size=batch_size, shuffle=True)

# Charger un batch d'images d'entrée (vous devez avoir des données d'entrée)
# Supposons que vous ayez un DataLoader nommé 'data_loader'
# Assurez-vous d'avoir défini 'data_loader' dans ce script ou de le charger à partir d'un autre endroit
# Créer un dossier pour sauvegarder les images reconstruites
output_dir = "images_reconstruites"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

global IMG_COORDS
IMG_COORDS = []
global PHOTOS
PHOTOS = []

# Fonction appelée lors du clic sur le bouton "Créer un portrait robot"
def create_portrait():
    for widget in center_frame.winfo_children():
         widget.destroy()

    # Proposer desimages décodées
    with torch.no_grad():
        input_batch = next(iter(data_loader))
    # Reconstruire les images à partir du modèle
    global recon_batch
    recon_batch, _, _ = autoencoder(input_batch)
    # Convertir les tenseurs PyTorch en numpy arrays
    input_batch = input_batch.numpy()
    recon_batch = recon_batch.detach().numpy()

    n = 10  # Nombre d'images à afficher
    # Enregistrer les images reconstruites au format JPG avec une correction des niveaux de gris
    for i in range(1, n + 1):
        a = recon_batch[i]-recon_batch[i].min()
        img_array = ((a/a.max())*255).astype(np.uint8)
        img = Image.fromarray(img_array.reshape(64, 64))
        img_path = os.path.join(output_dir, f"photo{i}.jpg")  # Utilisation de f-strings
        img.save(img_path, quality=90)
        IMG_COORDS.append(img)
    #print(len(IMG_COORDS))

    global selected_photos
    selected_photos = []
    global selected_photos_continue
    selected_photos_continue = []

    photo_list = []  # Liste pour stocker les objets PhotoImage

    # Charger les images reconstruites
    for i in range(1,n+1):
        img_path = os.path.join(output_dir, f"photo{i}.jpg")
        photo = ImageTk.PhotoImage(Image.open(img_path))
        photo_list.append(photo)

    def toggle_photo(photo_id):
        if photo_id in selected_photos:
            selected_photos.remove(photo_id)
        else:
            selected_photos.append(photo_id)

    photo_checkboxes = []

    # Affichage des photos dans la fenêtre principale
    photos_per_row = n // 2  # Nombre de photos par ligne
    row_count = (n + photos_per_row - 1) // photos_per_row  # Nombre total de lignes
    for row_index in range(row_count):
        row_frame = tk.Frame(center_frame)  # Créer un nouveau cadre pour chaque ligne
        row_frame.pack()  # Pack le cadre de la ligne
        for j in range(photos_per_row):
            i = row_index * photos_per_row + j  # Calcul de l'index de la photo
            if i >= n:  # Si nous avons dépassé le nombre total de photos, sortir de la boucle
                break
            photo = photo_list[i]
            photo_id = i+1
            checkbox_var = tk.BooleanVar()
            checkbox = tk.Checkbutton(row_frame, image=photo, variable=checkbox_var, onvalue=True, offvalue=False, command=lambda p=photo_id: toggle_photo(p))
            checkbox.image = photo
            checkbox.pack(side=tk.LEFT, padx=10, pady=10)
            photo_checkboxes.append(checkbox)

    # Suppression des éléments inutiles
    label_welcome.pack_forget()
    button_create.pack_forget()

    # Affichage des boutons "Continuer" et "Terminer"
    button_continue = tk.Button(center_frame, text="Continuer la sélection", command=continue_selection)
    button_continue.pack(side=tk.BOTTOM, padx=60, pady=10)
    #global button_panier
    button_panier = tk.Button(text="Photos sélectionnées", command=photos_selectioned)
    button_panier.place(relx=1.0, rely=0.0, anchor='ne', bordermode='outside', x=-30, y=150)
    button_newfaces = tk.Button(center_frame, text="Individus proposés", command=new_faces)
    button_newfaces.pack(side=tk.BOTTOM, padx=60, pady=10)
    #button_finish = tk.Button(center_frame, text="Terminer", command=finish_selection)
    #button_finish.pack(side=tk.BOTTOM, padx=60, pady=10)

def photos_selectioned():
    button_panier.destroy()
    for widget in center_frame.winfo_children():
         widget.destroy()
    label_explication = tk.Label(center_frame, text="Mes photos sélectionnées : ",bg="white",font=("Helvetica", 30))
    label_explication.pack(padx=20,pady=10,fill=tk.X)

    i=1
    for img in PHOTOS:
        img_path = os.path.join("images_selectionnees", f"photo{i}.jpg")  # Utilisation de f-strings
        img.save(img_path, quality=90)
        i+=1

    photo_list = []  # Liste pour stocker les objets PhotoImage
    n=len(PHOTOS)
    # Charger les images reconstruites
    for i in range(1,n+1):
        img_path = os.path.join("images_selectionnees", f"photo{i}.jpg")
        photo = ImageTk.PhotoImage(Image.open(img_path))
        photo_list.append(photo)

    photo_checkboxes = []
    # Affichage des photos dans la fenêtre principale
    if n<=1:
        photos_per_row = 1
        row_count = 1
    else :
        photos_per_row = n // 2  # Nombre de photos par ligne
        row_count = (n + photos_per_row - 1) // photos_per_row  # Nombre total de lignes
    for row_index in range(row_count):
        row_frame = tk.Frame(center_frame)  # Créer un nouveau cadre pour chaque ligne
        row_frame.pack()  # Pack le cadre de la ligne
        for j in range(photos_per_row):
            i = row_index * photos_per_row + j  # Calcul de l'index de la photo
            if i >= n:  # Si nous avons dépassé le nombre total de photos, sortir de la boucle
                break
            photo = photo_list[i]
            photo_id = i+1
            checkbox_var = tk.BooleanVar()
            checkbox = tk.Checkbutton(row_frame, image=photo, variable=checkbox_var, onvalue=True, offvalue=False, command=lambda p=photo_id: toggle_photo(p))
            checkbox.image = photo
            checkbox.pack(side=tk.LEFT, padx=10, pady=10)
            photo_checkboxes.append(checkbox)

    global button_retour
    button_retour = tk.Button(text="Retour", command=retour)
    button_retour.place(relx=1.0, rely=0.0, anchor='ne', bordermode='outside', x=-30, y=150)
    # Effacer le bouton button_panier
    if button_panier:
        button_panier.destroy()

def retour():
    button_retour.destroy()
    for widget in center_frame.winfo_children():
         widget.destroy()
         create_portrait()
    button_panier = tk.Button(text="Photos sélectionnées", command=photos_selectioned)
    button_panier.place(relx=1.0, rely=0.0, anchor='ne', bordermode='outside', x=-30, y=150)

def continue_selection():
    for widget in center_frame.winfo_children():
         widget.destroy()
    i=len(IMG_COORDS)-10
    for x in selected_photos:
        x+=i
        PHOTOS.append(IMG_COORDS[x-1])
    create_portrait()

# Fonction appelée lors du clic sur le bouton "Continuer"
def new_faces():
    global selected_photos_continue

    if len(selected_photos)==0 and len(selected_photos_continue)==0:
        label_comment = tk.Label(center_frame, text="Error : Veuillez sélectionner des photos", font=("Helvetica", 16), bg="red")
        label_comment.pack()
    else :
        for widget in center_frame.winfo_children():
             widget.destroy()
        # Fais quelque chose avec les paramètres du décodeur, par exemple :
        image_coords = [recon_batch[1,0,0].flatten(),recon_batch[2,0,0].flatten()]
        image_coords = np.array(image_coords)
        latent_coordinates = torch.tensor(image_coords)
        generated_image = autoencoder.decoder(latent_coordinates)
        image = generated_image.squeeze().detach().numpy()
        new_image_coords = algo_genetique.photos_methode_centroide(10,image_coords)
        new_images = []
        n = len(new_image_coords)
        for i in range(n):
            new_latent_coords = torch.tensor([new_image_coords[i]])
            new_latent_coords = new_latent_coords.float()
            new_gen_image = autoencoder.decoder(new_latent_coords)
            new_images.append(new_gen_image.squeeze().detach().numpy())
        
        # Créer un dossier pour sauvegarder les images reconstruites
        output_dir = "images_reconstruites"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        n = 10  # Nombre d'images à afficher
        for i in range(n):
            a = new_images[i]-new_images[i].min()
            img_array = ((a/a.max())*255).astype(np.uint8)
            img = Image.fromarray(img_array.reshape(64, 64))
            img_path = os.path.join(output_dir, f"photo{i}.jpg")  # Utilisation de f-strings
            img.save(img_path, quality=90)

        selected_photos_continue = []

        photo_list_continue = []  # Liste pour stocker les objets PhotoImage

        # Charger les images reconstruites
        for i in range(n):
            img_path = os.path.join(output_dir, f"photo{i}.jpg")
            photo = ImageTk.PhotoImage(Image.open(img_path))
            photo_list_continue.append(photo)

        def toggle_photo(photo_id):
            if photo_id in selected_photos_continue:
                selected_photos_continue.remove(photo_id)
            else:
                selected_photos_continue.append(photo_id)

        photo_checkboxes = []

        # Affichage des photos dans la fenêtre principale
        photos_per_row = n // 2  # Nombre de photos par ligne
        row_count = (n + photos_per_row - 1) // photos_per_row  # Nombre total de lignes
        for row_index in range(row_count):
            row_frame = tk.Frame(center_frame)  # Créer un nouveau cadre pour chaque ligne
            row_frame.pack()  # Pack le cadre de la ligne
            for j in range(photos_per_row):
                i = row_index * photos_per_row + j  # Calcul de l'index de la photo
                if i >= n:  # Si nous avons dépassé le nombre total de photos, sortir de la boucle
                    break
                photo = photo_list_continue[i]
                photo_id = i
                checkbox_var = tk.BooleanVar()
                checkbox = tk.Checkbutton(row_frame, image=photo, variable=checkbox_var, onvalue=True, offvalue=False, command=lambda p=photo_id: toggle_photo(p))
                checkbox.image = photo
                checkbox.pack(side=tk.LEFT, padx=10, pady=10)
                photo_checkboxes.append(checkbox)

        # Suppression des éléments inutiles
        label_welcome.pack_forget()
        button_create.pack_forget()

        # Affichage des boutons "Continuer" et "Terminer"
        button_newfaces = tk.Button(center_frame, text="Continuer", command=new_faces)
        button_newfaces.pack(side=tk.BOTTOM, padx=60, pady=10)
        button_finish = tk.Button(center_frame, text="Terminer", command=finish_selection)
        button_finish.pack(side=tk.BOTTOM, padx=60, pady=10)
        #create_portrait()


# Fonction appelée lors du clic sur le bouton "Terminer"
def finish_selection():
    if len(selected_photos)!=1 and len(selected_photos_continue)!=1:
        if len(selected_photos)==0 and len(selected_photos_continue)==0:
            label_comment = tk.Label(center_frame, text="Error : Veuillez sélectionner des photos", font=("Helvetica", 16), bg="red")
            label_comment.pack()
        if len(selected_photos)>1 or len(selected_photos_continue)>1:
            label_comment = tk.Label(center_frame, text="Error : Appuyer sur Continuer", font=("Helvetica", 16), bg="red")
            label_comment.pack()

    else :
        for widget in center_frame.winfo_children():
            widget.destroy()
        label_comment = tk.Label(center_frame, text="Voici le portrait robot final : ", font=("Helvetica", 20), bg="white")
        label_comment.pack()

        if len(selected_photos_continue)!=0:
            selected_photos[0]=selected_photos_continue[0]
        # Charger et afficher l'image
        img_path = os.path.join("images_reconstruites", f"photo{selected_photos[0]}"+'.jpg')
        photo_finale = ImageTk.PhotoImage(Image.open(img_path))
        #photo_finale = ImageTk.PhotoImage(Image.open(selected_photos[0]+'.jpg'))
        label_photo_final = tk.Label(center_frame, image=photo_finale)
        label_photo_final.image = photo_finale
        label_photo_final.pack()
        # Créer un bouton "Recommencer"
        button_back = tk.Button(center_frame, text="Recommencer", command=back_to_selection)
        button_back.pack()

def back_to_selection():
    for widget in center_frame.winfo_children():
         widget.destroy()
    label2_welcome = tk.Label(center_frame, text="Bienvenue dans le créateur de portraits robots !",bg="white",font=("Helvetica", 30))
    label2_welcome.pack(padx=20,pady=10,fill=tk.X)
    label2_explanation = tk.Label(center_frame, text=explanation_text, bg="white", font=("Helvetica", 14), justify="left")
    label2_explanation.pack(padx=20, pady=(0, 20))
    button2_create = tk.Button(center_frame, text="Créer un portrait robot", command=choose_method,foreground="black")#,font=("Helvetica", 15))
    button2_create.pack(pady=5)

# Fonction pour créer le cadre pour le titre et l'explication
def create_explanation_frame(method, explanation):
    frame = tk.Frame(center_frame, bg="white")
    frame.pack(pady=5, padx=20, fill=tk.X)

    # Titre en gras avec fond bleu foncé
    title_frame = tk.Frame(frame, bg="#000080", highlightbackground="#000080", highlightthickness=5)
    title_frame.pack(fill=tk.X)  # Utilisation de pack au lieu de grid
    title_text = tk.Text(title_frame, height=1, width=160, wrap=tk.WORD, background="#000080", borderwidth=0, fg="white")
    title_text.tag_configure("bold", font=("Helvetica", 12, "bold"))
    title_text.insert(tk.END, method + "\n", "bold")
    title_text.configure(state="disabled")
    title_text.pack(side="left")

    # Explication avec indentation négative pour aligner les points avec le début de chaque ligne du titre
    explanation_text = tk.Text(frame, height=8, width=160, wrap=tk.WORD, background="white", borderwidth=0)
    explanation_text.pack(fill=tk.X, padx=(0, 20))  # Utilisation de pack avec une marge droite
    explanation = "    " + explanation.replace("\n", "\n    ")  # Ajouter l'indentation négative
    explanation_text.insert(tk.END, explanation)
    explanation_text.configure(state="disabled")

    return frame  # Retourner le cadre pour permettre l'ajout de la case à cocher


# Définir une fonction pour choisir la méthode
def choose_method():
    # Détruire tous les widgets du cadre central
    for widget in center_frame.winfo_children():
        widget.destroy()

    checkboxes = []  # Liste pour stocker les cases à cocher

    # Afficher les explications de chaque méthode
    for i, method in enumerate(methods):
        # Créer le cadre pour le titre et l'explication
        explanation_frame = create_explanation_frame(method, explanations[method])

        # Créer la case à cocher et l'ajouter à la liste
        checkbox = tk.Checkbutton(explanation_frame, variable=method_var, onvalue=method, offvalue="", bg="white")
        checkbox.pack(side="left", padx=(20 if i != 0 else 0))  # Ajouter une marge gauche sauf pour le premier
        checkboxes.append(checkbox)
    # Affichage du bouton "Valider"
    button_continue = tk.Button(center_frame, text="Valider", command=validate_method)
    button_continue.pack(side=tk.BOTTOM, padx=60, pady=10)

def validate_method():
    global picked_method  # Déclarer la variable globale
    # Détruire tous les widgets du cadre central
    for widget in center_frame.winfo_children():
        widget.destroy()
    # Créer une étiquette pour afficher les erreurs
    error_label = tk.Label(center_frame, text="", font=("Helvetica", 16), bg="white")
    error_label.pack()
    selected_method = method_var.get()
    if selected_method:
        # Vérifier que seule une méthode a été choisie
        if selected_method.count(",") == 0:
            # Commencer la création de portrait robot en fonction de la méthode choisie
            if selected_method == "Méthode 1":
                picked_method = algo_genetique.photos_methode_centroide
                create_portrait()
            elif selected_method == "Méthode 2":
                picked_method = algo_genetique.photos_methode_crossover
                create_portrait()
            elif selected_method == "Méthode 3":
                picked_method = algo_genetique.photos_methode_noise
                create_portrait()
        else:
            # Afficher un message d'erreur si plus d'une méthode a été choisie
            error_label.config(text="Erreur : Veuillez sélectionner une seule méthode.", fg="red")
    else:
        # Afficher un message d'erreur si aucune méthode a été choisie
        error_label.config(text="Erreur : Veuillez sélectionner une méthode.", fg="red")
        # Redémarrer la sélection de méthode après un court délai
        center_frame.after(2000, choose_method)


# Création de la fenêtre principale
root = tk.Tk()
root.title("https://mon-portrait-robot.com")

# Récupère les dimensions de l'écran
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Définition de la taille de la fenêtre principale
root.geometry(f"{screen_width}x{screen_height}")
root.configure(bg="white")

# Création d'un cadre en haut de la fenêtre pour le titre
title_frame = tk.Frame(root,bg="#000080",height=100)
title_frame.pack(side=tk.TOP, pady=00, fill=tk.X)
title_frame.pack_propagate(0) #pour redéfinir les dimensions et être sur qu'elles soient bien prises en compte

# Ajouter un titre au cadre
label_title = tk.Label(title_frame, text="Créateur de portraits robots", font=("Helvetica", 50), foreground="white", bg="#000080",height=50)
label_title.pack()

# Ajout d'un cadre bleu sur le bord droit de l'écran
#leftside_frame = tk.Frame(root, bg="#000080", width=200, height=screen_height)
#leftside_frame.pack(fill=tk.Y, pady=0, side=tk.LEFT)
#leftside_frame.pack_propagate(0)

# Création d'un cadre pour la zone centrale
center_frame = tk.Frame(root,bg="white",bd=5)
center_frame.pack(padx=50, pady=50, expand=True)  # Définit l'expansion et le remplissage autour du cadre central

# Définition des composants de l'interface
label_welcome = tk.Label(center_frame, text="Bienvenue dans le créateur de portraits robots !",bg="white",font=("Helvetica", 30), anchor="center")
label_welcome.pack(padx=20,pady=10,fill=tk.X)

# Créer un bouton pour créer un portrait robot
button_create = tk.Button(center_frame, text="Créer un portrait robot", command=create_portrait,foreground="black")#,font=("Helvetica", 15))
button_create.pack(pady=25)
button_create.pack_forget()  # Masquer le bouton initialement

button_panier = tk.Button(text="Photos sélectionnées", command=photos_selectioned)

#button_method = tk.Button(center_frame, text="Méthodes : ", command=choose_method,foreground="black")#,font=("Helvetica", 15))
#button_method.pack(pady=5)


label2_welcome = tk.Label(center_frame, text="Bienvenue dans le créateur de portraits robots !",bg="white",font=("Helvetica", 30))
button2_create = tk.Button(center_frame, text="Créer un portrait robot", command=create_portrait,foreground="black")#,font=("Helvetica", 15))

# Ajout des explications sur l'objectif de l'application
explanation_text = """
Cette application vous permet de créer des portraits robots en utilisant un algorithme génétique.
Vous pouvez sélectionner une ou plusieurs photos de personnes, puis notre algorithme génétique va modifier
les visages pour obtenir un portrait robot qui ressemble le plus possible aux personnes que vous avez choisies.
Le but est d'obtenir le portrait robot le plus ressemblant possible afin de retrouver un potentiel coupable.
"""

label_explanation = tk.Label(center_frame, text=explanation_text, bg="white", font=("Helvetica", 14), justify="left")
label_explanation.pack(padx=20, pady=(0, 20))


# Définir les noms des méthodes et leurs explications
methods = ["Méthode 1", "Méthode 2", "Méthode 3"]
explanations = {
    "Méthode 1": """\
    Méthode 1 : Calcul des coordonnées du centroïde et génération de nouvelles photos.
        • Cette méthode fonctionne en calculant d'abord le vecteur de coordonnées des centroïdes des vecteurs fournis.
        • Elle parcourt donc la liste des vecteurs fournis et additionne les coordonnées de chaque vecteur à un vecteur centroïde initialisé à 0.
        • Ensuite, elle divise chaque coordonnée par le nombre total de vecteurs pour obtenir la moyenne.
        • Ce vecteur de coordonnées du centroïde représente donc le centre géométrique des vecteurs fournis.
        • Une fois le vecteur de coordonnées du centroïde calculé, la méthode génère une population de nouveaux vecteurs.
        • Ces nouveaux vecteurs sont créés autour du centroïde calculé afin de générer des photos similaires mais légèrement différentes.
        • Utile pour créer une variété de nouvelles photos basées sur un ensemble de photos initiales en conservant des caractères communs.""",

    "Méthode 2": """\
    Méthode 2 : Génération de nouveaux vecteurs par crossover aléatoire.
        • Cette méthode fonctionne en créant d'abord un nouveau vecteur composé de coodrdonnées de tous les vecteurs sélectionnés aléatoirement.
        • Pour chaque coordonnée du nouveau vecteur, la méthode sélectionne aléatoirement une coordonnée parmi celles de tous les vecteurs d'origine.
        • Une fois le nouveau vecteur composé, la méthode génère une population de nouveaux vecteurs en utilisant ce vecteur comme base.
        • Ces nouveaux vecteurs sont créés avec des variations aléatoires autour du vecteur initial.
        • Utile pour créer une variété de nouvelles photos en combinant de manière aléatoire les caractéristiques des photos initiales.""",

    "Méthode 3": """\
    Méthode 3 : Introduction de bruit dans les vecteurs avant génération.
        • Cette méthode consiste tout d'abord à appliquer du bruit à chacun des vecteurs fournis.
        • Pour cela, elle ajoute un bruit aléatoire à chaque coordonnée de chaque vecteur.
        • Ensuite, elle génère une population de nouveaux vecteurs à partir des vecteurs bruités.
        • Pour chaque vecteur bruité, la méthode crée un nouveau vecteur en ajoutant un peu de bruit supplémentaire à chaque coordonnée.
        • Ces nouveaux vecteurs conservent les caractéristiques des vecteurs d'origine mais présentent des variations dues au bruit introduit.
        • Utile pour créer une variété de photos en introduisant des variations aléatoires mais contrôlées dans les caractéristiques des photos initiales."""
}


# Créer une variable pour stocker la méthode choisie
method_var = tk.StringVar()

button_method = tk.Button(center_frame, text="Choisir une méthode", command=choose_method,foreground="black")#,font=("Helvetica", 15))
button_method.pack(pady=5)

# Chargement de l'image du logo
#logo_image = Image.open("logo.png")  # Remplacez "logo.png" par le chemin de votre fichier logo
#logo_photo = ImageTk.PhotoImage(logo_image)

# Création d'un label pour afficher le logo
#logo_label = tk.Label(root, image=logo_photo, bg="white")
#logo_label.pack(side=tk.RIGHT, padx=10, pady=10, anchor="nw")  # Placer le logo dans le coin en haut à gauche


# Lancement de la boucle principale de l'interface graphique
root.mainloop()