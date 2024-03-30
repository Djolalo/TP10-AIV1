import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

image_GT = cv2.imread('./Fichiers_utiles_TP10_2023/2classes_100_100_8bits_GT.png',
                      cv2.IMREAD_GRAYSCALE)
image_2 = cv2.imread('./Fichiers_utiles_TP10_2023/2classes_100_100_8bits.png', cv2.IMREAD_GRAYSCALE)

def seuilOpti(img, parf):
    init = np.array(img)

    histo = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(histo)
    plt.show()
    seuil_optimal = 0
    prob_max = -np.inf
    total = img.size
    for t1 in range(256):
        N1 = np.sum(histo[0:t1])
        P1 = N1 / total

        P2 = 1 - P1
        N2 = total - N1
        noccu1=0 # nombre d'occurences pondéré par le niveau de gris dans omega1
        noccu2 = 0 #nombre d'occurences pondéré par le niveau de gris dans omega2
        for i in range(256):
            if i < t1 :
                noccu1 += histo[i] * i
            else:
                noccu2 += histo[i]*i
        print(noccu1+noccu2)
        mu1 = np.divide(noccu1,N1,  where=(N1!=0))
        mu2 = np.divide(noccu2,N2,  where=(N2!=0))
        #print(f'N1 = {N1}\t N2 = {N2}\t noccu1  = {noccu1}\t noccu2 = {noccu2}\t mu1 = {mu1}\t mu2 = {mu2}')
        prob_courante = P1 * P2 * (mu1-mu2)**2 if mu1 != 0 and mu2 != 0 else 0
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