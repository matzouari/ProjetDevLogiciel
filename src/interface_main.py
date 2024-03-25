import tkinter as tk
from PIL import ImageTk, Image

import torch

<<<<<<< HEAD:interface_main.py
# with torch.no_grad():
#     input_batch = next(iter(data_loader))
#
# # Reconstruire les images à partir du modèle
# #global recon_batch
# recon_batch, _, _ = autoencoder(input_batch)
#
# # Convertir les tenseurs PyTorch en numpy arrays
# input_batch = input_batch.numpy()
# recon_batch = recon_batch.detach().numpy()

# # Supprimer les images après utilisation (facultatif)
# for i in range(n):
#     img_path = os.path.join(output_dir, f"reconstructed_image_{i}.jpg")
#     os.remove(img_path)

global IMG_COORDS
IMG_COORDS = []
global PHOTOS
PHOTOS = []
#print(len(PHOTOS))
#print(len(IMG_COORDS))
=======
from AlgoGenetique import algo_genetique
from VAE import autoencodeur_celebA as autoencodeur
from VAE.autoencodeur_celebA import data_loader

import os
import numpy as np
>>>>>>> 750be73b96706bd66e48e0c7978430e0ce77c6fc:src/interface_main.py

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
    global button_panier
    button_panier = tk.Button(text="Photos sélectionnées", command=photos_selectioned)
    button_panier.place(relx=1.0, rely=0.0, anchor='ne', bordermode='outside', x=-30, y=150)
    button_newfaces = tk.Button(center_frame, text="Individus proposés", command=new_faces)
    button_newfaces.pack(side=tk.BOTTOM, padx=60, pady=10)
    #button_finish = tk.Button(center_frame, text="Terminer", command=finish_selection)
    #button_finish.pack(side=tk.BOTTOM, padx=60, pady=10)

def photos_selectioned():
    for widget in center_frame.winfo_children():
         widget.destroy()
    label_explication = tk.Label(center_frame, text="Mes photos sélectionnées : ",bg="white",font=("Helvetica", 30))
    label_explication.pack(padx=20,pady=10,fill=tk.X)

<<<<<<< HEAD:interface_main.py
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
    button_panier = tk.Button(text="Photos sélectionnées", command=photos_selectioned)
    button_panier.place(relx=1.0, rely=0.0, anchor='ne', bordermode='outside', x=-30, y=150)
    # Affichage des boutons "Continuer" et "Terminer"
    #button_continue = tk.Button(center_frame, text="Continuer", command=continue_selection)
    #button_finish = tk.Button(center_frame, text="Terminer", command=finish_selection)
    #button_continue.pack(side=tk.BOTTOM, padx=60, pady=10)
    #button_finish.pack(side=tk.BOTTOM, padx=60, pady=10)

=======
# Fonction appelée lors du clic sur le bouton "Continuer"
>>>>>>> 750be73b96706bd66e48e0c7978430e0ce77c6fc:src/interface_main.py
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

    #print(selected_photos)
    #print(selected_photos_continue)

    if len(selected_photos)==0 and len(selected_photos_continue)==0:
        label_comment = tk.Label(center_frame, text="Error : Veuillez sélectionner des photos", font=("Helvetica", 16), bg="red")
        label_comment.pack()
    else :
        for widget in center_frame.winfo_children():
             widget.destroy()
<<<<<<< HEAD:interface_main.py
        # Fais quelque chose avec les paramètres du décodeur, par exemple :
        image_coords = [recon_batch[1,0,0].flatten(),recon_batch[2,0,0].flatten()]
=======
        image_coords = [recon_batch[1,0,0].flatten()]
>>>>>>> 750be73b96706bd66e48e0c7978430e0ce77c6fc:src/interface_main.py
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
        #for i in range(n):
            # Afficher les images reconstruites
            #ax = plt.subplot(2, n, i + 1 + n)
            #plt.imshow(new_images[i], cmap='gray')
            #plt.title('Image Reconstruite')
            #plt.axis('off')
        #plt.show()
        # Créer un dossier pour sauvegarder les images reconstruites
        output_dir = "images_reconstruites"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        n = 10  # Nombre d'images à afficher
        # Enregistrer les images reconstruites au format JPG avec une correction des niveaux de gris
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
    # Insère ici la logique pour afficher la photo sélectionnée en grand
     # Afficher le commentaire au-dessus de la photo

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

# Fonction appelée lors du clic sur le bouton "Retour"
def back_to_selection():
    for widget in center_frame.winfo_children():
         widget.destroy()
    label2_welcome = tk.Label(center_frame, text="Bienvenue dans le créateur de portraits robots !",bg="white",font=("Helvetica", 30))
    label2_welcome.pack(padx=20,pady=10,fill=tk.X)
    button2_create = tk.Button(center_frame, text="Créer un portrait robot", command=create_portrait,foreground="black")#,font=("Helvetica", 15))
    button2_create.pack(pady=5)
    #create_portrait()

<<<<<<< HEAD:interface_main.py
def method():
    for widget in center_frame.winfo_children():
         widget.destroy()
    label_explication = tk.Label(center_frame, text="Explications : ",bg="white",font=("Helvetica", 30))
    label_explication.pack(padx=20,pady=10,fill=tk.X)
=======
# Définir les dimensions
latent_dim = 64

# Charger le modèle sauvegardé
checkpoint = torch.load("src/VAE/vae_model.pth")
autoencoder = autoencodeur.VAE(latent_dim)
autoencoder.load_state_dict(checkpoint)
autoencoder.eval()  # Mettre le modèle en mode évaluation

# Charger un batch d'images d'entrée (vous devez avoir des données d'entrée)
# Supposons que vous ayez un DataLoader nommé 'data_loader'
# Assurez-vous d'avoir défini 'data_loader' dans ce script ou de le charger à partir d'un autre endroit
# Créer un dossier pour sauvegarder les images reconstruites
output_dir = "images_reconstruites"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# # Supprimer les images après utilisation (facultatif)
# for i in range(n):
#     img_path = os.path.join(output_dir, f"reconstructed_image_{i}.jpg")
#     os.remove(img_path)
>>>>>>> 750be73b96706bd66e48e0c7978430e0ce77c6fc:src/interface_main.py

# Création de la fenêtre principale
root = tk.Tk()
root.title("https://CriminAI.com")

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

button_method = tk.Button(center_frame, text="Méthodes : ", command=method,foreground="black")#,font=("Helvetica", 15))
button_method.pack(pady=5)


label2_welcome = tk.Label(center_frame, text="Bienvenue dans le créateur de portraits robots !",bg="white",font=("Helvetica", 30))
button2_create = tk.Button(center_frame, text="Créer un portrait robot", command=create_portrait,foreground="black")#,font=("Helvetica", 15))

# Chargement de l'image du logo
#logo_image = Image.open("logo.png")  # Remplacez "logo.png" par le chemin de votre fichier logo
#logo_photo = ImageTk.PhotoImage(logo_image)

# Création d'un label pour afficher le logo
#logo_label = tk.Label(root, image=logo_photo, bg="white")
#logo_label.pack(side=tk.RIGHT, padx=10, pady=10, anchor="nw")  # Placer le logo dans le coin en haut à gauche

# Lancement de la boucle principale de l'interface graphique
root.mainloop()
