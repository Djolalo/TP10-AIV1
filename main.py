import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

image_GT = cv2.imread('./Fichiers_utiles_TP10_2023/2classes_100_100_8bits_GT.png',
                      cv2.IMREAD_GRAYSCALE)
image_2 = cv2.imread('./Fichiers_utiles_TP10_2023/2classes_100_100_8bits.png', cv2.IMREAD_GRAYSCALE)

def seuilOpti(img, parf):
    init = np.array(img)

    histo = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten().astype(int)
    plt.plot(histo)
    plt.show()
    seuil_optimal = 0
    prob_max = -np.inf
    total = img.size
    for t1 in range(256):
        N1 = np.sum(histo[:t1])
        P1 = N1 / total
        mu1 = 0 if N1 == 0 else np.sum(np.arange(0,t1) * histo[0: t1]) / N1 # calcul de la moyenne 

        P2 = 1 - P1
        N2 = total - N1
        mu2 = 0 if N2 == 0 else np.sum(np.arange(t1+1, 256) * histo[t1+1:]) / N2
        #print(f'N1 = {N1}\t N2 = {N2}\t noccu1  = {noccu1}\t noccu2 = {noccu2}\t mu1 = {mu1}\t mu2 = {mu2}')
        prob_courante = P1 * P2 * (mu1-mu2)**2 
        if prob_courante > prob_max:
            seuil_optimal = t1
            prob_max = prob_courante

    print("Seuil optimal:", seuil_optimal)

    '''
    GT_1D = parf.flatten()
    dest_1D = dest.flatten()
    cm = confusion_matrix(GT_1D, dest_1D)
    accuracy = (np.diag(cm).sum() / img.size) * 100  # Pour avoir en %
    print(f"Taux de bonne classification des 3 classes: {accuracy}%")
    '''


seuilOpti(image_2, image_GT)