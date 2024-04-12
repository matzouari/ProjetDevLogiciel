import tkinter as tk
from PIL import ImageTk, Image
import torch
from torch.utils.data import DataLoader
from utils.AlgoGenetique import algo_genetique
from src.utils.VAE import autoencodeur_BW as autoencodeur
from src.utils.VAE.autoencodeur_BW import data_loader
import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import random
import os
import numpy as np

# Définir les dimensions
latent_dim = 64

# Charger le modèle sauvegardé
checkpoint = torch.load("src/interface/models/vae_model.pth")
autoencoder = autoencodeur.VAE(latent_dim)
autoencoder.load_state_dict(checkpoint)
autoencoder.eval()  # Mettre le modèle en mode évaluation

# Charger un batch d'images d'entrée (vous devez avoir des données d'entrée)
# Supposons que vous ayez un DataLoader nommé 'data_loader'
# Assurez-vous d'avoir défini 'data_loader' dans ce script ou de le charger à partir d'un autre endroit
# Créer un dossier pour sauvegarder les images reconstruites
output_dir = "src/interface/images_selectionnees"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# # Supprimer les images après utilisation (facultatif)
# for i in range(n):
#     img_path = os.path.join(output_dir, f"reconstructed_image_{i}.jpg")
#     os.remove(img_path)

global IMG_COORDS
IMG_COORDS = []
global PHOTOS
PHOTOS = []
global PHOTOS_ID
PHOTOS_ID = []


# Créer une variable pour suivre si le bouton a été créé
button_panier_created = False

# Définir la fonction pour créer le bouton
def create_button_panier():
    global button_panier_created
    if not button_panier_created:
        global button_selection
        button_selection = tk.Button(text="Photos sélectionnées", fg='black',command=photos_selectioned)
        button_selection.place(relx=1.0, rely=0.0, anchor='ne', bordermode='outside', x=-30, y=150)
        button_panier_created = True

# Définir la fonction pour supprimer le bouton
def destroy_button_panier():
    global button_panier_created
    if button_panier_created:
        button_selection.destroy()
        button_panier_created = False

# Fonction appelée lors du clic sur le bouton "Créer un portrait robot"
def create_portrait():
    for widget in center_frame.winfo_children():
         widget.destroy()
    for widget in low_frame.winfo_children():
         widget.destroy()

    photos_selectioned_page_principale()

    label_explication = tk.Label(center_frame,
                                text="Commencez par sélectionner les photos qui se rapprochent le plus de l'individu que vous pensez avoir aperçu. Appuyez sur 'Continuer la sélection' pour que d'autres photos soient proposées. Une fois votre sélection finie, appuyez sur 'Lancer l'algorithme génétique' pour que l'algorithme génétique utilise les informations des images que vous avez choisi, pour vous proposer des protraits robots ressemblants.",
                                bg="white",
                                fg='black',
                                font=("Helvetica", 18),
                                anchor="n",
                                wraplength=850)
    label_explication.pack(padx=20,pady=10,fill=tk.X)

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

    global selected_photos
    selected_photos = []

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
    button_continue = tk.Button(center_frame, text="Continuer la sélection", fg='black',command=continue_selection)
    button_continue.pack(side=tk.BOTTOM, padx=60, pady=10)

    if len(PHOTOS)!=0:
        button_newfaces = tk.Button(center_frame, text="Lancer l'algorithme génétique", fg='black',command=new_faces)
        button_newfaces.pack(side=tk.BOTTOM, padx=60, pady=10)
        create_button_panier()


def photos_selectioned():
    destroy_button_panier()
    for widget in center_frame.winfo_children():
         widget.destroy()
    for widget in low_frame.winfo_children():
         widget.destroy()
    label_explication = tk.Label(center_frame,
                                text="Voici les photos que vous avez sélectionnées pour le moment. Si vous pensez vous être trompé vous pouvez enlever une ou plusieurs photos en les sélectionnant puis en appuyant sur le bouton 'Enlever de la sélection'.",
                                bg="white",
                                fg='black',
                                font=("Helvetica", 18),
                                anchor="n",
                                wraplength=850)
    label_explication.pack(padx=20,pady=10,fill=tk.X)

    def toggle_photo(photo_id):
        if photo_id in selected_photos:
            selected_photos.remove(photo_id)
        else:
            selected_photos.append(photo_id)


    def enlever_photos_selectionnees():
        global PHOTOS
        PHOTOS = [PHOTOS[i] for i in range(len(PHOTOS)) if i+1 not in selected_photos]
        photos_selectioned()

    i=1
    for img in PHOTOS:
        img_path = os.path.join("src/interface/images_selectionnees", f"photo{i}.jpg")  # Utilisation de f-strings
        img.save(img_path, quality=90)
        i+=1

    photo_list = []  # Liste pour stocker les objets PhotoImage
    n=len(PHOTOS)
    # Charger les images reconstruites
    for i in range(1,n+1):
        img_path = os.path.join("src/interface/images_selectionnees", f"photo{i}.jpg")
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
            # Ajouter l'image à la liste des photos sélectionnées si elle est cochée
            #checkbox_var.trace_add('write', lambda name, index, mode, var=checkbox_var, img=img: update_selected_photos(var, img))

    global button_retour
    button_retour = tk.Button(text="Retour", command=retour)
    button_retour.place(relx=1.0, rely=0.0, anchor='ne', bordermode='outside', x=-30, y=150)
    # Ajouter un bouton pour enlever les photos sélectionnées
    button_enlever = tk.Button(center_frame, text="Enlever de la sélection", fg='black',command=enlever_photos_selectionnees)
    button_enlever.pack(padx=10, pady=10)

def retour():
    button_retour.destroy()
    for widget in center_frame.winfo_children():
         widget.destroy()
         create_portrait()

def continue_selection():
    for widget in center_frame.winfo_children():
         widget.destroy()
    i=len(IMG_COORDS)-10
    for x in selected_photos:
        PHOTOS_ID.append(x)
        x+=i
        PHOTOS.append(IMG_COORDS[x-1])
    create_portrait()


def photos_selectioned_page_principale():
    #global low_frame
    for widget in center_frame.winfo_children() :
         widget.destroy()
    for x in low_frame.winfo_children() :
        x.destroy()

    if PHOTOS :

        label_explication = tk.Label(low_frame,
                                    text="Les photos que vous avez sélectionnées :",
                                    bg="white",
                                    fg='black',
                                    font=("Helvetica", 18),
                                    anchor="nw",
                                    wraplength=850)
        label_explication.pack(padx=20,pady=10,fill=tk.X)
        i=1
        for img in PHOTOS:
            img_path = os.path.join("src/interface/images_selectionnees", f"photo{i}.jpg")  # Utilisation de f-strings
            img.save(img_path, quality=90)
            i+=1

        photo_list = []  # Liste pour stocker les objets PhotoImage
        n=len(PHOTOS)
        # Charger les images reconstruites
        for i in range(1,n+1):
            img_path = os.path.join("src/interface/images_selectionnees", f"photo{i}.jpg")
            photo = ImageTk.PhotoImage(Image.open(img_path))
            photo_list.append(photo)

        # Affichage des photos dans la fenêtre principale
        for photo in photo_list:
            label_photo = tk.Label(low_frame, image=photo)
            label_photo.image = photo  # Garde une référence à l'objet PhotoImage
            label_photo.pack(side=tk.LEFT, padx=10, pady=10)



#### Partie ALGO GENETIQUE ####
# Fonction appelée lors du clic sur le bouton "Continuer"
def new_faces():
    destroy_button_panier()
    photos_selectioned_page_principale()

    global selected_photos_continue
    for widget in center_frame.winfo_children():
            widget.destroy()

    label_explication = tk.Label(center_frame,
                                text="Les images présentées ci-dessous sont des portraits robots qui ont été reconstruits à partir des informations récupérées lors de l'étape précédente. Vous pouvez désormais sélectionner un ou plusieurs portraits robots. Le bouton 'Continuer' permet de présenter de nouveaux portraits robots. En revanche, pour pouvoir terminer cette étape vous devez appuyer sur le bouton 'Terminer' en ayant sélectionner un seul portrait robot, celui se rapporchant le plus de l'individu recherché.",
                                bg="white",
                                fg='black',
                                font=("Helvetica", 18),
                                anchor="n",
                                wraplength=850)
    label_explication.pack(padx=20,pady=10,fill=tk.X)

    image_coords=[]
    for x in PHOTOS_ID :
        image_coords.append(recon_batch[x,0,0].flatten())
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
    output_dir = "src/interface/images_reconstruites"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n = 10  # Nombre d'images à afficher
    # Enregistrer les images reconstruites au format JPG avec une correction des niveaux de gris
    for i in range(1,n+1):
        a = new_images[i-1]-new_images[i-1].min()
        img_array = ((a/a.max())*255).astype(np.uint8)
        img = Image.fromarray(img_array.reshape(64, 64))
        img_path = os.path.join(output_dir, f"photo{i}.jpg") 
        img.save(img_path, quality=90)
        IMG_COORDS.append(img)

    selected_photos_continue = []

    photo_list_continue = []  # Liste pour stocker les objets PhotoImage

    # Charger les images reconstruites
    for i in range(1,n+1):
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
    button_newfaces = tk.Button(center_frame, text="Continuer", fg='black',command=continue_selection_bis)
    button_newfaces.pack(side=tk.BOTTOM, padx=60, pady=10)
    button_finish = tk.Button(center_frame, text="Terminer",fg='black', command=finish_selection)
    button_finish.pack(side=tk.BOTTOM, padx=60, pady=10)
    create_button_panier_bis()


button_panier_created_bis = False

# Définir la fonction pour créer le bouton
def create_button_panier_bis():
    global button_panier_created_bis
    if not button_panier_created_bis:
        global button_selection_bis
        button_selection_bis = tk.Button(text="Photos sélectionnées", fg='black',command=photos_selectioned_bis)
        button_selection_bis.place(relx=1.0, rely=0.0, anchor='ne', bordermode='outside', x=-30, y=150)
        button_panier_created_bis = True

# Définir la fonction pour supprimer le bouton
def destroy_button_panier_bis():
    global button_panier_created_bis
    if button_panier_created_bis:
        button_selection_bis.destroy()
        button_panier_created_bis = False

def photos_selectioned_bis():

    destroy_button_panier_bis()

    for widget in center_frame.winfo_children():
         widget.destroy()
    for widget in low_frame.winfo_children():
         widget.destroy()
    label_explication = tk.Label(center_frame,
                                text="Voici les photos que vous avez sélectionnées pour le moment. Si vous pensez vous être trompé vous pouvez enlever une ou plusieurs photos en les sélectionnant puis en appuyant sur le bouton 'Enlever de la sélection'.",
                                bg="white",
                                fg='black',
                                font=("Helvetica", 18),
                                anchor="n",
                                wraplength=850)
    label_explication.pack(padx=20,pady=10,fill=tk.X)

    def toggle_photo(photo_id):
        if photo_id in selected_photos:
            selected_photos.remove(photo_id)
        else:
            selected_photos.append(photo_id)


    def enlever_photos_selectionnees():
        global PHOTOS
        PHOTOS = [PHOTOS[i] for i in range(len(PHOTOS)) if i+1 not in selected_photos]
        photos_selectioned_bis()

    i=1
    for img in PHOTOS:
        img_path = os.path.join("src/interface/images_reconstruites", f"photo{i}.jpg")  # Utilisation de f-strings
        img.save(img_path, quality=90)
        i+=1

    photo_list = []  # Liste pour stocker les objets PhotoImage
    n=len(PHOTOS)
    # Charger les images reconstruites
    for i in range(1,n+1):
        img_path = os.path.join("src/interface/images_reconstruites", f"photo{i}.jpg")
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
    button_retour = tk.Button(text="Retour", fg='black',command=retour_bis)
    button_retour.place(relx=1.0, rely=0.0, anchor='ne', bordermode='outside', x=-30, y=150)
    button_enlever = tk.Button(center_frame, text="Enlever de la sélection", fg='black',command=enlever_photos_selectionnees)
    button_enlever.pack(padx=10, pady=10)

def retour_bis():
    button_retour.destroy()
    for widget in center_frame.winfo_children():
         widget.destroy()
         new_faces()

def continue_selection_bis():
    for widget in center_frame.winfo_children():
         widget.destroy()
    i=len(IMG_COORDS)-10
    for x in selected_photos_continue:
        x+=i
        PHOTOS.append(IMG_COORDS[x-1])
    new_faces()


#### PAGE DE FIN ####
# Fonction appelée lors du clic sur le bouton "Terminer"
def finish_selection():
    destroy_button_panier_bis()
    create_button_panier_p3()
    # Insère ici la logique pour afficher la photo sélectionnée en grand
     # Afficher le commentaire au-dessus de la photo
    if len(selected_photos)!=1 and len(selected_photos_continue)!=1:
        if len(selected_photos)==0 and len(selected_photos_continue)==0:
            label_comment = tk.Label(center_frame, text="Error : Veuillez sélectionner le portrait robot qui ressemble le plus a l'individu recherché pour Terminer.", font=("Helvetica", 16), fg='black',bg="red")
            label_comment.pack()
        if len(selected_photos)>1 or len(selected_photos_continue)>1:
            label_comment = tk.Label(center_frame, text="Error : Pour Terminer veuillez sélectionner une seule image, sinon veuillez appuyer sur Continuer", font=("Helvetica", 16), fg='black',bg="red")
            label_comment.pack()
    else :
        for widget in center_frame.winfo_children():
            widget.destroy()
        for widget in low_frame.winfo_children():
            widget.destroy()
        photos_recap()
        label_comment = tk.Label(final_frame, text="Voici le portrait robot final : ", font=("Helvetica", 20), fg='black',bg="white")
        label_comment.pack(padx=50, pady=10)

        if len(selected_photos_continue)!=0:
            selected_photos.append(0)
            selected_photos[0]=selected_photos_continue[0]
        # Charger et afficher l'image
        img_path = os.path.join("src/interface/images_reconstruites", f"photo{selected_photos[0]}"+'.jpg')
        photo_finale = ImageTk.PhotoImage(Image.open(img_path))
        #photo_finale = ImageTk.PhotoImage(Image.open(selected_photos[0]+'.jpg'))
        label_photo_final = tk.Label(final_frame, image=photo_finale)
        label_photo_final.image = photo_finale
        label_photo_final.pack()
        # Créer un bouton "Recommencer"
        button_back = tk.Button(final_frame, text="Recommencer",fg='black', command=back_to_selection)
        button_back.pack(pady=20)


def photos_recap():
    label_explication = tk.Label(center_frame,
                                text="Votre démarche était de retrouver un individu et pour cela vous avez procédé à la réalisation d'un portrait robot. Au cours du processus vous avez sélectionné plusiers images correspondant à des individus qui ressemblent à celui qui est recherché. Vous pouvez appuyer sur 'Recommencer' pour refaire tout le processus de reconnaissance, que vous venez de réaliser.",
                                bg="white",
                                fg='black',
                                font=("Helvetica", 18),
                                anchor="nw",
                                wraplength=850)
    label_explication.pack(padx=10,pady=10,fill=tk.X)

    i=1
    for img in PHOTOS:
        img_path = os.path.join("src/interface/images_reconstruites", f"photo{i}.jpg")  # Utilisation de f-strings
        img.save(img_path, quality=90)
        i+=1

    photo_list = []  # Liste pour stocker les objets PhotoImage
    n=len(PHOTOS)
    # Charger les images reconstruites
    for i in range(1,n+1):
        img_path = os.path.join("src/interface/images_reconstruites", f"photo{i}.jpg")
        photo = ImageTk.PhotoImage(Image.open(img_path))
        photo_list.append(photo)

    # Affichage des photos dans la fenêtre principale
    for photo in photo_list:
        label_photo = tk.Label(center_frame, image=photo)
        label_photo.image = photo  # Garde une référence à l'objet PhotoImage
        label_photo.pack(side=tk.LEFT, padx=20, pady=10)


button_panier_created_p3 = False

# Définir la fonction pour créer le bouton
def create_button_panier_p3():
    global button_panier_created_p3
    if not button_panier_created_p3:
        global button_selection_p3
        button_selection_p3 = tk.Button(text="Photos sélectionnées", fg='black',command=photos_selectioned_p3)
        button_selection_p3.place(relx=1.0, rely=0.0, anchor='ne', bordermode='outside', x=-30, y=150)
        button_panier_created_p3 = True

# Définir la fonction pour supprimer le bouton
def destroy_button_panier_p3():
    global button_panier_created_p3
    if button_panier_created_p3:
        button_selection_p3.destroy()
        button_panier_created_p3 = False

def photos_selectioned_p3():
    destroy_button_panier_p3()
    for widget in center_frame.winfo_children():
         widget.destroy()
    for widget in low_frame.winfo_children():
         widget.destroy()
    for widget in final_frame.winfo_children():
         widget.destroy()

    label_explication = tk.Label(center_frame,
                                text="Voici les photos que vous avez sélectionnées pour le moment. Si vous pensez vous être trompé vous pouvez enlever une ou plusieurs photos en les sélectionnant puis en appuyant sur le bouton 'Enlever de la sélection'.",
                                bg="white",
                                fg='black',
                                font=("Helvetica", 18),
                                anchor="n",
                                wraplength=850)
    label_explication.pack(padx=20,pady=10,fill=tk.X)

    def toggle_photo(photo_id):
        if photo_id in selected_photos:
            selected_photos.remove(photo_id)
        else:
            selected_photos.append(photo_id)


    def enlever_photos_selectionnees():
        global PHOTOS
        PHOTOS = [PHOTOS[i] for i in range(len(PHOTOS)) if i+1 not in selected_photos]
        photos_selectioned_p3()

    i=1
    for img in PHOTOS:
        img_path = os.path.join("src/interface/images_reconstruites", f"photo{i}.jpg")  # Utilisation de f-strings
        img.save(img_path, quality=90)
        i+=1

    photo_list = []  # Liste pour stocker les objets PhotoImage
    n=len(PHOTOS)
    # Charger les images reconstruites
    for i in range(1,n+1):
        img_path = os.path.join("src/interface/images_reconstruites", f"photo{i}.jpg")
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
    button_retour = tk.Button(text="Retour", command=retour_p3)
    button_retour.place(relx=1.0, rely=0.0, anchor='ne', bordermode='outside', x=-30, y=150)
    button_enlever = tk.Button(center_frame, text="Enlever de la sélection", command=enlever_photos_selectionnees)
    button_enlever.pack(padx=10, pady=10)

def retour_p3():
    button_retour.destroy()
    for widget in center_frame.winfo_children():
         widget.destroy()
    finish_selection()



#### POUR RECOMMENCER ####
# Fonction appelée lors du clic sur le bouton "Retour"  : c plus le cas, j'ai court circuité le truc pour directement choisir une méthode (toujours la même ou changer) après avoir appuyé sur retour
def back_to_selection():
    destroy_button_panier_p3()
    global IMG_COORDS
    IMG_COORDS = []
    global PHOTOS
    PHOTOS = []
    global PHOTOS_ID
    PHOTOS_ID = []
    for widget in center_frame.winfo_children():
         widget.destroy()
    for widget in root.winfo_children():
        if isinstance(widget, tk.Button) and widget['text'] == "Retour":
            widget.destroy()
    #for widget in low_frame.winfo_children():
         #widget.destroy()
    for widget in final_frame.winfo_children():
         widget.destroy()

    # Supprimer les images après utilisation (facultatif)
    #for i in range(1,11):
        #img_path = os.path.join("images_reconstruites", f"photo{i}.jpg")
        #os.remove(img_path)
    label2_welcome = tk.Label(center_frame, text="Bienvenue dans le créateur de portraits robots !",fg='black',bg="white",font=("Helvetica", 30))
    label2_welcome.pack(padx=20,pady=10,fill=tk.X)
    label2_explanation = tk.Label(center_frame, text=explanation_text, fg='black', bg="white", font=("Helvetica", 14), justify="left")
    label2_explanation.pack(padx=20, pady=(0, 20))
    button2_create = tk.Button(center_frame, text="Créer un portrait robot", command=choose_method,fg='black',foreground="black")#,font=("Helvetica", 15))
    button2_create.pack(pady=5)


#### PAGE METHODES ####
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
    explanation_text = tk.Text(frame, height=8, width=160, wrap=tk.WORD, background="white",fg='black', borderwidth=0)
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
        checkbox = tk.Checkbutton(explanation_frame, variable=method_var, onvalue=method, offvalue="",fg='black', bg="white")
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
    error_label = tk.Label(center_frame, text="", font=("Helvetica", 16),fg='black',bg="white")
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
label_title = tk.Label(title_frame, text="Créateur de portraits robots", font=("Helvetica", 50), foreground="white",fg='black',bg="#000080",height=50)
label_title.pack()

# Création d'un cadre pour la zone centrale
center_frame = tk.Frame(root,bg="white",bd=5)
center_frame.pack(padx=50, pady=50, expand=True)  # Définit l'expansion et le remplissage autour du cadre central

final_frame = tk.Frame(root, bg="white", bd=5)
final_frame.pack(side=tk.TOP, anchor='n')

# Création d'un cadre pour la zone du bas
low_frame = tk.Frame(root,bg="white",bd=5)
low_frame.pack(side=tk.LEFT, anchor=tk.S)

# Définition des composants de l'interface
label_welcome = tk.Label(center_frame, text="Bienvenue dans le créateur de portraits robots !",fg='black',bg="white",font=("Helvetica", 30), anchor="center")
label_welcome.pack(padx=20,pady=10,fill=tk.X)

# Créer un bouton pour créer un portrait robot
button_create = tk.Button(center_frame, text="Créer un portrait robot", command=create_portrait,fg='black',foreground="black")#,font=("Helvetica", 15))
button_create.pack(pady=25)
button_create.pack_forget()  # Masquer le bouton initialement

# Créer un bouton pour accéder au panier es photos sélectionnées
#button_panier = tk.Button(text="Photos sélectionnées", command=photos_selectioned)
#button_panier_after_retour = tk.Button(text="Photos sélectionnées", command=photos_selectioned)
# Créer une Frame pour contenir le bouton
#button_frame = tk.Frame(root, bg="white")
#button_frame.pack(side=tk.TOP, anchor='ne', padx=10, pady=10)
#button_panier_p2= tk.Button(text="Photos sélectionnées 2", command=photos_selectioned_p2)

# Créer un bouton pour créer un portrait robot dans une autres fonction
label2_welcome = tk.Label(center_frame, text="Bienvenue dans le créateur de portraits robots !",fg='black',bg="white",font=("Helvetica", 30))
button2_create = tk.Button(center_frame, text="Créer un portrait robot", command=create_portrait,fg='black',foreground="black")#,font=("Helvetica", 15))

# Ajout des explications sur l'objectif de l'application
explanation_text = """
Cette application vous permet de créer des portraits robots en utilisant un algorithme génétique.
Vous pouvez sélectionner une ou plusieurs photos de personnes, puis notre algorithme génétique va modifier
les visages pour obtenir un portrait robot qui ressemble le plus possible aux personnes que vous avez choisies.
Le but est d'obtenir le portrait robot le plus ressemblant possible afin de retrouver un potentiel coupable.
"""

label_explanation = tk.Label(center_frame, text=explanation_text,fg='black', bg="white", font=("Helvetica", 14), justify="left")
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

button_method = tk.Button(center_frame, text="Choisir une méthode", command=choose_method,fg='black',foreground="black")#,font=("Helvetica", 15))
button_method.pack(pady=5)


# Lancement de la boucle principale de l'interface graphique
root.mainloop()
