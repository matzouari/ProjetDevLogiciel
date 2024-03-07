import tkinter as tk
from PIL import ImageTk, Image  # Assure-toi d'avoir installé le module Pillow (PIL)


# Fonction appelée lors du clic sur le bouton "Créer un portrait robot"
def create_portrait():
    for widget in center_frame.winfo_children():
         widget.destroy()
    # Supposons que la fonction génère trois images, nous allons simplement créer des libellés pour les afficher
    # Remplace cette logique avec ta propre logique de génération d'images
    leftside_frame.pack_forget()
    #label2_welcome.destroy()
    #button2_create.pack_forget()
    logo_label.pack_forget()

    global selected_photos
    selected_photos = []

    photo1 = ImageTk.PhotoImage(Image.open("photo1.jpg"))
    photo2 = ImageTk.PhotoImage(Image.open("photo2.jpg"))
    photo3 = ImageTk.PhotoImage(Image.open("photo3.jpg"))

    def toggle_photo(photo_id):
        if photo_id in selected_photos:
            selected_photos.remove(photo_id)
        else:
            selected_photos.append(photo_id)

    photo_checkboxes = []

    # Affichage des photos dans la fenêtre principale
    for photo_id, photo in zip(["photo1", "photo2", "photo3"], [photo1, photo2, photo3]):
        checkbox_var = tk.BooleanVar()
        checkbox = tk.Checkbutton(center_frame, image=photo, variable=checkbox_var, onvalue=True, offvalue=False, command=lambda p=photo_id: toggle_photo(p))
        checkbox.image = photo
        checkbox.pack(side=tk.LEFT, padx=10)
        photo_checkboxes.append(checkbox)

    # Suppression des éléments inutiles
    label_welcome.pack_forget()
    button_create.pack_forget()

    # Affichage des boutons "Continuer" et "Terminer"
    button_continue = tk.Button(center_frame, text="Continuer", command=continue_selection)
    button_finish = tk.Button(center_frame, text="Terminer", command=finish_selection)
    button_continue.pack(side=tk.BOTTOM, padx=60, pady=10)
    button_finish.pack(side=tk.BOTTOM, padx=60, pady=10)


# Fonction appelée lors du clic sur le bouton "Continuer"
def continue_selection():
    # for widget in center_frame.winfo_children():
    #     widget.destroy()
    # create_portrait()
    if len(selected_photos)==0:
        label_comment = tk.Label(center_frame, text="Error : Veuillez sélectionner des photos", font=("Helvetica", 16), bg="red")
        label_comment.pack()
    else :
        for widget in center_frame.winfo_children():
             widget.destroy()
        create_portrait()


# Fonction appelée lors du clic sur le bouton "Terminer"
def finish_selection():
    # Insère ici la logique pour afficher la photo sélectionnée en grand
     # Afficher le commentaire au-dessus de la photo
    if len(selected_photos)!=1:
        if len(selected_photos)==0:
            label_comment = tk.Label(center_frame, text="Error : Veuillez sélectionner des photos", font=("Helvetica", 16), bg="red")
            label_comment.pack()
        if len(selected_photos)>1:
            label_comment = tk.Label(center_frame, text="Error : Appuyer sur Continuer", font=("Helvetica", 16), bg="red")
            label_comment.pack()
        # Créer un bouton "Retour"
        #button_back = tk.Button(center_frame, text="Retour", command=back_to_selection)
        #button_back.pack()
    else :
        for widget in center_frame.winfo_children():
            widget.destroy()
        label_comment = tk.Label(center_frame, text="Voici le portrait robot final : ", font=("Helvetica", 20), bg="white")
        label_comment.pack()

        # Charger et afficher l'image
        photo_finale = ImageTk.PhotoImage(Image.open(selected_photos[0]+'.jpg'))
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

# Création de la fenêtre principale
root = tk.Tk()
root.title("https://mon-protrait-robot.com")

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
leftside_frame = tk.Frame(root, bg="#000080", width=200, height=screen_height)
leftside_frame.pack(fill=tk.Y, pady=0, side=tk.LEFT)
leftside_frame.pack_propagate(0)

# Création d'un cadre pour la zone centrale
center_frame = tk.Frame(root,bg="white",bd=5)
center_frame.pack(padx=50, pady=50, expand=True)  # Définit l'expansion et le remplissage autour du cadre central

# Définition des composants de l'interface
label_welcome = tk.Label(center_frame, text="Bienvenue dans le créateur de portraits robots !",bg="white",font=("Helvetica", 30), anchor="center")
label_welcome.pack(padx=20,pady=10,fill=tk.X)

# Créer un bouton pour créer un portrait robot
button_create = tk.Button(center_frame, text="Créer un portrait robot", command=create_portrait,foreground="black")#,font=("Helvetica", 15))
button_create.pack(pady=5)

label2_welcome = tk.Label(center_frame, text="Bienvenue dans le créateur de portraits robots !",bg="white",font=("Helvetica", 30))
button2_create = tk.Button(center_frame, text="Créer un portrait robot", command=create_portrait,foreground="black")#,font=("Helvetica", 15))

# Chargement de l'image du logo
logo_image = Image.open("logo.png")  # Remplacez "logo.png" par le chemin de votre fichier logo
logo_photo = ImageTk.PhotoImage(logo_image)

# Création d'un label pour afficher le logo
logo_label = tk.Label(root, image=logo_photo, bg="white")
logo_label.pack(side=tk.RIGHT, padx=10, pady=10, anchor="nw")  # Placer le logo dans le coin en haut à gauche


# Lancement de la boucle principale de l'interface graphique
root.mainloop()
