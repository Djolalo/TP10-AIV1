import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

image_GT = cv2.imread('./Fichiers_utiles_TP10_2023/3classes_100_156_8bits_GT.png',
                      cv2.IMREAD_GRAYSCALE)
image_2 = cv2.imread('./Fichiers_utiles_TP10_2023/3classes_100_156_8bits.png', cv2.IMREAD_GRAYSCALE)

def seuilOpti(img, parf, show = True):
    init = np.array(img)
    '''
    On caste l'histo en entier pour éviter les erreurs d'arrondis, et on l'applatit.
    Cela facilite les manipulations avec numpy. (l'histo rendu par openCV est un tableau 2d (peu pratique...))
    '''
    histo = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten().astype(int) 
    if show: 
        plt.plot(histo)
        plt.show()
    seuil_optimal = 0 , 0 
    prob_max = np.inf
    total = img.size
    for t1 in range(256):
        N1 = np.sum(histo[:t1])
        P1 = N1 / total

        mu1 = 0 if N1 == 0 else np.sum(np.arange(t1) * histo[: t1]) / N1 # calcul de la moyenne 
        variance1 = 0 if N1 == 0 else np.sum(np.power(np.arange(t1) -mu1,2)*histo[:t1]) / N1 
        for t2 in range(t1+1,256): 
            N2 = np.sum(histo[t1+1: t2])
            P2 = N2 / total

            N3 = total - N2 - N1 
            P3 = 1 - P2 - P1
            '''
             Calcul des moyennes des niveaux de gris de chaque classe
             On pondère le nombre d'occurences à un dit niveau de gris par la valeur du niveau de gris
             Le tout rapporté à l'effectif total
             On calcule la classe 1 en dehors de la boucle pour économiser des calculs
            '''
            mu2 = 0 if N2 == 0 else np.sum(np.arange(t1+1, t2) * histo[t1+1: t2]) / N2
            mu3 = 0 if N3 == 0 else np.sum(np.arange(t2+1, 256) * histo[t2+1:]) / N3
            '''
             Calcul des variances des niveaux de gris au sein de chaque classe
             On pondère le nombre d'occurences à un dit niveau de gris par la distance entre le niveau de gris et la moyenne
             Le tout rapporté à l'effectif total
             On calcule la classe 1 en dehors de la boucle pour économiser des calculs
            '''
            variance2 = 0 if N2 == 0 else np.sum(np.power(np.arange(t1+1, t2) -mu2,2)*histo[t1+1:t2]) / N2
            variance3 = 0 if N3 == 0 else np.sum(np.power(np.arange(t2+1, 256) -mu3,2)*histo[t2+1:]) / N3
            '''
             on calcule la proba d'être dans telle classe multipliée par sa variance.
             Cela nous permet d'estimer la probabilité qu'elle déborde sur une autre
            '''
            prob_courante = (P1 * variance1) + (P2 * variance2) + (P3 * variance3) 
            if prob_courante < prob_max: #on minimise la variance de chaque classe, ce qui nous permettra d'éviter qu'une courbe en superpose une autre
                seuil_optimal = t1, t2
                prob_max = prob_courante

    print("Seuil optimal:", seuil_optimal)

    _, img_seuil_1 = cv2.threshold(img, seuil_optimal[0], 127, cv2.THRESH_BINARY)
    _, img_seuil_2 = cv2.threshold(img, seuil_optimal[1], 255, cv2.THRESH_BINARY)
    dest = np.maximum(img_seuil_1, img_seuil_2)
    if show: 
        cv2.namedWindow("Image finale", cv2.WINDOW_NORMAL)
        cv2.imshow("Image finale", dest)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    GT_1D = parf.flatten()
    dest_1D = dest.flatten()
    cm = confusion_matrix(GT_1D, dest_1D)
    accuracy = (np.diag(cm).sum() / img.size) * 100  # Pour avoir en %
    print(f"Taux de bonne classification des 3 classes: {accuracy}%")



seuilOpti(image_2, image_GT)